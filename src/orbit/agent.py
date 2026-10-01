"""The experiment loop: a small tree search over candidate solutions.

Every node is one experiment (a script + its official score). Each step picks one of:
  draft    write a fresh solution (until `n_drafts` exist, or when nothing works yet)
  debug    repair a failed node (bounded by `max_debug_depth` per lineage)
  improve  make one focused change to the best node (occasionally a runner-up, to explore)
"""
import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from . import prompts
from .llm import LLM
from .metrics import (InvalidPredictions, check_values, combine_predictions, read_predictions,
                      score)
from .sandbox import SCRIPT, ExecResult, Sandbox
from .task import Task

ERROR_CHARS = 4_000
log = logging.getLogger("orbit")


@dataclass
class SearchConfig:
    max_steps: int = 20
    time_budget: float = 3600.0        # seconds for the whole search
    token_budget: int | None = None    # total prompt + completion tokens
    n_drafts: int = 3
    max_debug_depth: int = 3
    debug_prob: float = 0.5
    explore_prob: float = 0.2
    exec_timeout: int = 600
    cpus: float = 4
    seed: int = 0


@dataclass
class Node:
    id: int
    stage: str
    parent: int | None
    plan: str = ""
    code: str | None = None
    metric: float | None = None        # official validation score (mean over folds)
    fold_scores: list[float] = field(default_factory=list)  # one score per validation fold
    test_metric: float | None = None   # filled for the final best node only
    error: str | None = None
    duration: float = 0.0
    debug_depth: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def buggy(self) -> bool:
        return self.metric is None


@dataclass
class Journal:
    nodes: list[Node] = field(default_factory=list)
    best_id: int | None = None
    stop_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def get(self, node_id: int) -> Node:
        return self.nodes[node_id - 1]

    def good(self) -> list[Node]:
        return [n for n in self.nodes if not n.buggy]

    def ranked(self, metric) -> list[Node]:
        return sorted(self.good(), key=lambda n: n.metric, reverse=metric.greater_is_better)

    def has_children(self, node: Node) -> bool:
        return any(n.parent == node.id for n in self.nodes)

    @property
    def tokens(self) -> int:
        return sum(n.prompt_tokens + n.completion_tokens for n in self.nodes)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Journal":
        d = json.loads(Path(path).read_text())
        d["nodes"] = [Node(**n) for n in d["nodes"]]
        return cls(**d)


class Agent:
    def __init__(self, task: Task, llm: LLM, sandbox: Sandbox, config: SearchConfig,
                 tracker=None, on_node: Callable[[Node, "Agent"], None] | None = None):
        self.task, self.llm, self.sandbox, self.cfg = task, llm, sandbox, config
        self.tracker = tracker
        self.on_node = on_node or (lambda node, agent: None)
        self.rng = random.Random(config.seed)
        self.journal = Journal()
        self.system = prompts.system_prompt(task, config.exec_timeout, config.cpus)
        self._ids = None

    # ---- policy -------------------------------------------------------------------------
    def next_action(self) -> tuple[str, Node | None]:
        j, cfg = self.journal, self.cfg
        drafts = sum(n.stage == "draft" for n in j.nodes)
        debuggable = [n for n in j.nodes if n.buggy and n.code is not None
                      and n.debug_depth < cfg.max_debug_depth and not j.has_children(n)]
        good = j.ranked(self.task.metric)
        if drafts < cfg.n_drafts:
            return "draft", None
        if debuggable and (not good or self.rng.random() < cfg.debug_prob):
            return "debug", debuggable[-1]
        if not good:
            return "draft", None
        if len(good) > 1 and self.rng.random() < cfg.explore_prob:
            return "improve", self.rng.choice(good[1:3])
        return "improve", good[0]

    def build_prompt(self, stage: str, parent: Node | None) -> str:
        data = prompts.data_section(self.task)
        history = prompts.history_section(self.journal.nodes)
        if stage == "draft":
            return prompts.DRAFT.format(data=data, history=history)
        if stage == "improve":
            return prompts.IMPROVE.format(data=data, history=history, node_id=parent.id,
                                          score=parent.metric, plan=parent.plan, code=parent.code)
        return prompts.DEBUG.format(data=data, node_id=parent.id, plan=parent.plan,
                                    code=parent.code, error=parent.error)

    # ---- one experiment -----------------------------------------------------------------
    def step(self) -> Node:
        stage, parent = self.next_action()
        node = Node(id=len(self.journal.nodes) + 1, stage=stage,
                    parent=parent.id if parent else None,
                    debug_depth=parent.debug_depth + 1 if stage == "debug" else 0)
        user_prompt = self.build_prompt(stage, parent)
        reply = self.llm.complete(self.system, user_prompt)
        node.prompt_tokens, node.completion_tokens = reply.prompt_tokens, reply.completion_tokens
        node.plan, node.code = prompts.parse_response(reply.text)

        workdir = self.task.run_dir / "nodes" / f"{node.id:03d}"
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "prompt.md").write_text(f"# System\n\n{self.system}\n\n# User\n\n{user_prompt}",
                                           encoding="utf-8")
        (workdir / "response.md").write_text(reply.text, encoding="utf-8")

        if node.code is None:
            node.error = ("The reply was cut off inside a code block (output token limit?)."
                          if reply.text.count("```") % 2 else
                          "The reply contained no ```python code block.")
        else:
            self._run_folds(node, workdir)

        self.journal.nodes.append(node)
        best = self.best()
        self.journal.best_id = best.id if best else None
        self.journal.save(self.task.run_dir / "journal.json")
        if self.tracker:
            self.tracker.log_node(node, workdir)
        self.on_node(node, self)
        return node

    def _fwork(self, workdir: Path, fold: int) -> Path:
        return workdir if self.task.n_folds == 1 else workdir / f"fold_{fold}"

    def _run_folds(self, node: Node, workdir: Path) -> None:
        """Run the script once per fold and score each fold's validation rows."""
        task, n = self.task, self.task.n_folds
        out, err, scores = [], [], []
        for k in range(1, n + 1):
            fwork = self._fwork(workdir, k)
            result = self.sandbox.run(node.code, fwork, task.fold_dir(k), self.cfg.exec_timeout)
            node.duration += result.duration
            tag = f"fold {k}/{n}: " if n > 1 else ""
            if n > 1:
                out.append(f"=== fold {k}/{n} ===\n{result.stdout}")
                err.append(f"=== fold {k}/{n} ===\n{result.stderr}")
            else:
                out, err = [result.stdout], [result.stderr]
            error, value = self._score_fold(result, fwork, k)
            if error:
                node.error = tag + error
                break
            scores.append(value)
        (workdir / "stdout.txt").write_text("\n".join(out), encoding="utf-8")
        (workdir / "stderr.txt").write_text("\n".join(err), encoding="utf-8")
        if n > 1:
            (workdir / SCRIPT).write_text(node.code, encoding="utf-8")
        if not node.error:
            node.fold_scores = scores
            node.metric = sum(scores) / len(scores)

    def _score_fold(self, result: ExecResult, fwork: Path, fold: int) -> tuple[str | None, float | None]:
        if not result.ok:
            return (result.stderr or result.stdout or f"exit code {result.returncode}")[-ERROR_CHARS:], None
        try:
            val = self.task.labels("val", fold)
            preds = read_predictions(fwork / "predictions.csv", self.task.fold_ids(fold))
            # Check test rows too, so the final test scoring cannot fail on them.
            check_values(self.task.metric, preds, self.task.positive_label)
            return None, score(self.task.metric, val, preds.loc[val.index], self.task.positive_label)
        except InvalidPredictions as e:
            return f"Script ran but its output was rejected: {e}", None

    def best(self) -> Node | None:
        ranked = self.journal.ranked(self.task.metric)
        return ranked[0] if ranked else None

    # ---- the loop -----------------------------------------------------------------------
    def _budget_exhausted(self) -> str | None:
        j, cfg = self.journal, self.cfg
        if len(j.nodes) >= cfg.max_steps:
            return f"step budget ({cfg.max_steps}) reached"
        if time.time() - j.started_at >= cfg.time_budget:
            return f"time budget ({cfg.time_budget:.0f}s) reached"
        if cfg.token_budget and j.tokens >= cfg.token_budget:
            return f"token budget ({cfg.token_budget:,}) reached"
        return None

    def run(self) -> Journal:
        llm_failures = 0
        while not (reason := self._budget_exhausted()):
            try:
                self.step()
                llm_failures = 0
            except KeyboardInterrupt:
                reason = "interrupted by user"
                break
            except Exception as e:  # noqa: BLE001 - LLM/network hiccups shouldn't lose the run
                llm_failures += 1
                if llm_failures >= 3:
                    reason = f"stopped after repeated errors: {type(e).__name__}: {e}"
                    break
                time.sleep(2 * llm_failures)
        self.finalize(reason)
        return self.journal

    def finalize(self, reason: str) -> None:
        """Score the best experiment once on the untouched test split."""
        j = self.journal
        j.stop_reason, j.finished_at = reason, time.time()
        best = self.best()
        if best:
            test = self.task.labels("test")
            workdir = self.task.run_dir / "nodes" / f"{best.id:03d}"
            try:
                per_fold = [read_predictions(self._fwork(workdir, k) / "predictions.csv",
                                             self.task.fold_ids(k)).loc[test.index]
                            for k in range(1, self.task.n_folds + 1)]
                preds = combine_predictions(self.task.metric, per_fold)
                best.test_metric = score(self.task.metric, test, preds, self.task.positive_label)
            except Exception as e:  # noqa: BLE001 - still save the journal and the report
                log.warning("could not score experiment #%s on the test split: %s", best.id, e)
            j.best_id = best.id
        j.save(self.task.run_dir / "journal.json")
        if self.tracker:
            self.tracker.finish(self.task, j)
