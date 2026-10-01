"""Task setup: split data, hide labels, profile the dataset for the agent.

Data layout inside a run directory:

    input/train.csv   features + target           (mounted read-only into the sandbox)
    input/test.csv    features + orbit_id, no labels (validation and test rows, shuffled together)
    hidden/labels.csv orbit_id, split, target     (never mounted; used only by the scorer)

The agent cannot tell which unlabeled rows are validation (used to rank experiments
during search) and which are test (scored once, for the final best experiment).
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .metrics import ID_COL, METRICS, Metric, score

MAX_CLASSES = 30


@dataclass
class TaskSpec:
    data_path: str
    target: str
    metric: str | None = None
    goal: str | None = None
    val_size: float = 0.2
    test_size: float = 0.2
    seed: int = 42


@dataclass
class Task:
    spec: TaskSpec
    run_dir: Path
    kind: str
    metric_name: str
    positive_label: str | None
    classes: list
    n_train: int
    profile: str
    baseline: float

    @property
    def metric(self) -> Metric:
        return METRICS[self.metric_name]

    @property
    def input_dir(self) -> Path:
        return self.run_dir / "input"

    def labels(self, split: str) -> pd.Series:
        df = pd.read_csv(self.run_dir / "hidden" / "labels.csv")
        df = df[df["split"] == split].set_index(ID_COL)
        return df["target"]

    def save(self) -> None:
        d = asdict(self)
        d["run_dir"] = str(self.run_dir)
        (self.run_dir / "task.json").write_text(json.dumps(d, indent=2, default=str))

    @classmethod
    def load(cls, run_dir: Path) -> "Task":
        d = json.loads((Path(run_dir) / "task.json").read_text())
        d["spec"] = TaskSpec(**d["spec"])
        d["run_dir"] = Path(run_dir)
        return cls(**d)


def infer_kind(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= MAX_CLASSES:
        return "classification"
    return "regression"


def default_metric(kind: str, n_classes: int) -> str:
    if kind == "regression":
        return "rmse"
    return "roc_auc" if n_classes == 2 else "accuracy"


def profile(df: pd.DataFrame, target: str, max_cols: int = 60) -> str:
    """Compact, LLM-friendly description of the training data."""
    lines = [f"Rows: {len(df):,}  Columns: {df.shape[1]} (including target {target!r})", "",
             "column | dtype | missing% | unique | example values"]
    cols = [c for c in df.columns if c != target][:max_cols]
    for c in [target] + cols:
        s = df[c]
        examples = ", ".join(map(str, s.dropna().unique()[:4]))
        tag = "  <- TARGET" if c == target else ""
        lines.append(f"{c} | {s.dtype} | {100 * s.isna().mean():.1f} | {s.nunique()} | "
                     f"{examples[:80]}{tag}")
    if df.shape[1] - 1 > max_cols:
        lines.append(f"... and {df.shape[1] - 1 - max_cols} more columns")
    y = df[target]
    lines.append("")
    if infer_kind(y) == "classification":
        dist = y.value_counts(normalize=True).head(10)
        lines.append("Target distribution: " + ", ".join(f"{k}: {v:.1%}" for k, v in dist.items()))
    else:
        lines.append(f"Target stats: mean {y.mean():.4g}, std {y.std():.4g}, "
                     f"min {y.min():.4g}, max {y.max():.4g}")
    return "\n".join(lines)


def _baseline(metric: Metric, y_train: pd.Series, y_val: pd.Series, positive_label) -> float:
    """Score of a model that ignores the features, so results have a reference point."""
    if metric.needs_proba:
        prior = float((y_train.astype(str) == str(positive_label)).mean())
        preds = pd.Series(prior, index=y_val.index)
    elif metric.kind == "classification":
        preds = pd.Series(y_train.astype(str).mode()[0], index=y_val.index)
    else:
        preds = pd.Series(float(y_train.mean()), index=y_val.index)
    return score(metric, y_val, preds, positive_label)


def prepare(spec: TaskSpec, run_dir: Path) -> Task:
    df = pd.read_csv(spec.data_path)
    if spec.target not in df.columns:
        raise ValueError(f"Target column {spec.target!r} not found. Columns: {list(df.columns)}")
    if ID_COL in df.columns:
        raise ValueError(f"Column name {ID_COL!r} is reserved by orbit; please rename it.")
    df = df[df[spec.target].notna()].reset_index(drop=True)

    y = df[spec.target]
    kind = infer_kind(y)
    classes = sorted(map(str, y.unique())) if kind == "classification" else []
    metric_name = spec.metric or default_metric(kind, len(classes))
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric {metric_name!r}. Choose from {sorted(METRICS)}.")
    metric = METRICS[metric_name]
    if metric.kind != kind:
        raise ValueError(f"Metric {metric_name!r} is for {metric.kind}, but the target looks like {kind}.")
    if metric.needs_proba and len(classes) != 2:
        raise ValueError(f"{metric_name} is supported for binary targets only; try accuracy or f1_macro.")
    # Positive class for probability metrics: the minority class (usually the event of interest).
    positive_label = str(y.astype(str).value_counts().idxmin()) if metric.needs_proba else None

    strat = y.astype(str) if kind == "classification" else None
    holdout = spec.val_size + spec.test_size
    train, rest = train_test_split(df, test_size=holdout, stratify=strat, random_state=spec.seed)
    strat_rest = rest[spec.target].astype(str) if kind == "classification" else None
    val, test = train_test_split(rest, test_size=spec.test_size / holdout,
                                 stratify=strat_rest, random_state=spec.seed)

    hidden = pd.concat([val.assign(_split="val"), test.assign(_split="test")])
    hidden = hidden.sample(frac=1.0, random_state=spec.seed).reset_index(drop=True)
    rng = np.random.default_rng(spec.seed)
    hidden[ID_COL] = rng.permutation(len(hidden)) + 1

    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    (run_dir / "hidden").mkdir(parents=True, exist_ok=True)
    train.to_csv(run_dir / "input" / "train.csv", index=False)
    features = [c for c in df.columns if c != spec.target]
    hidden[[ID_COL] + features].to_csv(run_dir / "input" / "test.csv", index=False)
    hidden[[ID_COL, "_split", spec.target]].rename(
        columns={"_split": "split", spec.target: "target"}
    ).to_csv(run_dir / "hidden" / "labels.csv", index=False)

    task = Task(
        spec=spec, run_dir=run_dir, kind=kind, metric_name=metric_name,
        positive_label=positive_label, classes=classes, n_train=len(train),
        profile=profile(train, spec.target),
        baseline=_baseline(metric, train[spec.target], val[spec.target], positive_label),
    )
    task.save()
    return task
