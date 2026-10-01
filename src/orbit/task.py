"""Task setup: split data, hide labels, profile the dataset for the agent.

Data layout inside a run directory. With one validation split (`folds == 1`):

    input/train.csv   features + target           (mounted read-only into the sandbox)
    input/test.csv    features + orbit_id, no labels (validation and test rows, shuffled together)
    hidden/labels.csv orbit_id, split, fold, target  (never mounted; used only by the scorer)

With k-fold cross-validation (`folds > 1`) every fold gets its own pair of files, and only that
fold's directory is mounted, so a script never sees another fold's labels:

    input/fold_1/train.csv, input/fold_1/test.csv   ... input/fold_k/...

Each fold's test.csv holds that fold's validation rows plus all test rows. The agent cannot tell
which unlabeled rows are validation (used to rank experiments during search) and which are test
(scored once, for the final best experiment, from the average of the k fold predictions).
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .metrics import ID_COL, METRICS, Metric, score
from .planner import describe_table, read_docs, read_table

MAX_CLASSES = 30
AUTO_FOLDS = 5
AUTO_CV_MAX_ROWS = 10_000  # larger datasets use one validation split: k runs per experiment is costly


@dataclass
class TaskSpec:
    data_path: str
    target: str
    metric: str | None = None
    goal: str | None = None
    val_size: float = 0.2
    test_size: float = 0.2
    folds: int | None = None  # None: 5-fold CV for small datasets, one split otherwise; 1: one split
    seed: int = 42
    sources: list[str] = field(default_factory=list)  # what the user pointed at (files, dirs, URLs)
    eval_text: str | None = None                      # the evaluation wording the user gave
    extras: list[str] = field(default_factory=list)   # additional tables, readable under data/extra/
    docs: list[str] = field(default_factory=list)     # documentation files shown to the model
    id_column: str | None = None                      # identifies rows; not a feature
    notes: list[str] = field(default_factory=list)    # how the task was set up


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
    n_folds: int = 1
    extra_profile: str = ""   # description of the additional tables
    docs_text: str = ""       # documentation shown to the model

    @property
    def metric(self) -> Metric:
        return METRICS[self.metric_name]

    @property
    def input_dir(self) -> Path:
        return self.run_dir / "input"

    def fold_dir(self, fold: int) -> Path:
        """Directory mounted into the sandbox for fold `fold` (1-based)."""
        return self.input_dir if self.n_folds == 1 else self.input_dir / f"fold_{fold}"

    def _labels_df(self) -> pd.DataFrame:
        return pd.read_csv(self.run_dir / "hidden" / "labels.csv")

    def labels(self, split: str, fold: int | None = None) -> pd.Series:
        df = self._labels_df()
        df = df[df["split"] == split]
        if fold is not None and "fold" in df.columns:
            df = df[df["fold"] == fold]
        return df.set_index(ID_COL)["target"]

    def fold_ids(self, fold: int) -> pd.Index:
        """The rows a script must predict in fold `fold`: its validation rows plus the test rows."""
        df = self._labels_df()
        if "fold" not in df.columns:
            return pd.Index(df[ID_COL])
        return pd.Index(df.loc[(df["fold"] == fold) | (df["split"] == "test"), ID_COL])

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


def _split(df: pd.DataFrame, target: str, kind: str, size: float, seed: int):
    """Stratified split for classification; plain split when a class is too rare to stratify."""
    if kind == "classification":
        try:
            return train_test_split(df, test_size=size, stratify=df[target].astype(str),
                                    random_state=seed)
        except ValueError:
            pass
    return train_test_split(df, test_size=size, random_state=seed)


def prepare(spec: TaskSpec, run_dir: Path) -> Task:
    df = read_table(Path(spec.data_path))
    if spec.target not in df.columns:
        raise ValueError(f"Target column {spec.target!r} not found. Columns: {list(df.columns)}")
    if ID_COL in df.columns:
        raise ValueError(f"Column name {ID_COL!r} is reserved by orbit; please rename it.")
    df = df[df[spec.target].notna()].reset_index(drop=True)
    y = df[spec.target]
    if pd.api.types.is_float_dtype(y) and (y == y.round()).all():
        # Missing values make pandas read integer targets (e.g. 0/1) as floats; restore them.
        df[spec.target] = y = y.astype("int64")

    kind = infer_kind(y)
    classes = sorted(map(str, y.unique())) if kind == "classification" else []
    metric_name = spec.metric or default_metric(kind, len(classes))
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric {metric_name!r}. Choose from {sorted(METRICS)}.")
    metric = METRICS[metric_name]
    if metric.kind == "regression" and kind == "classification" and pd.api.types.is_numeric_dtype(y):
        # An explicit regression metric overrides the guess for small-integer targets (ratings, counts).
        kind, classes = "regression", []
    if metric.kind != kind:
        raise ValueError(f"Metric {metric_name!r} is for {metric.kind}, but the target looks like {kind}.")
    if metric.needs_nonneg and (pd.to_numeric(y, errors="coerce") < 0).any():
        raise ValueError(f"{metric_name} needs a target >= 0, but {spec.target!r} has negative values.")
    if metric.needs_proba and len(classes) != 2:
        raise ValueError(f"{metric_name} is supported for binary targets only; try accuracy or f1_macro.")
    # Positive class for probability metrics: the minority class (usually the event of interest).
    positive_label = str(y.astype(str).value_counts().idxmin()) if metric.needs_proba else None

    rng = np.random.default_rng(spec.seed)
    if spec.folds is not None and spec.folds < 1:
        raise ValueError("folds must be at least 1.")
    n_folds = spec.folds or (AUTO_FOLDS if len(df) * (1 - spec.test_size) <= AUTO_CV_MAX_ROWS else 1)

    if n_folds > 1 and kind == "classification":
        # Stratified folds need every class in every fold; the test split takes its share first.
        usable = int(int(y.astype(str).value_counts().min()) * (1 - spec.test_size))
        if usable < n_folds:
            reduced = usable if usable >= 2 else 1
            spec.notes.append(f"folds reduced from {n_folds} to {reduced}: the rarest class has "
                              f"only {int(y.astype(str).value_counts().min())} rows")
            n_folds = reduced

    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    (run_dir / "hidden").mkdir(parents=True, exist_ok=True)
    features = [c for c in df.columns if c != spec.target]

    if n_folds == 1:
        holdout = spec.val_size + spec.test_size
        train, rest = _split(df, spec.target, kind, holdout, spec.seed)
        val, test = _split(rest, spec.target, kind, spec.test_size / holdout, spec.seed)
        parts = [(train, val, test)]
        dev = train
        val_rows, test_rows = [val.assign(_fold=1)], test.assign(_fold=0)
        profile_df = train
    else:
        dev, test = _split(df, spec.target, kind, spec.test_size, spec.seed)
        if len(dev) < 2 * n_folds:
            raise ValueError(f"Too few rows ({len(dev)}) for {n_folds}-fold cross-validation; "
                             "pass --folds 1 or provide more data.")
        parts = []
        val_rows, test_rows = [], test.assign(_fold=0)
        for k, (tr_idx, va_idx) in enumerate(_folds(dev, spec.target, kind, n_folds, spec.seed), 1):
            parts.append((dev.iloc[tr_idx], dev.iloc[va_idx], test))
            val_rows.append(dev.iloc[va_idx].assign(_fold=k))
        profile_df = dev

    if metric.needs_proba:
        for _, val_k, _ in parts:
            if val_k[spec.target].astype(str).nunique() < 2:
                raise ValueError(
                    f"{metric_name} needs both classes in every validation fold, but there are too few "
                    f"rows of the rarer class ({int(y.astype(str).value_counts().min())}) to split; "
                    "use --metric accuracy or balanced_accuracy, or provide more data.")
    hidden = pd.concat([pd.concat(val_rows).assign(_split="val"), test_rows.assign(_split="test")])
    hidden = hidden.sample(frac=1.0, random_state=spec.seed).reset_index(drop=True)
    hidden[ID_COL] = rng.permutation(len(hidden)) + 1
    hidden[[ID_COL, "_split", "_fold", spec.target]].rename(
        columns={"_split": "split", "_fold": "fold", spec.target: "target"}
    ).to_csv(run_dir / "hidden" / "labels.csv", index=False)

    extras = _read_extras(spec.extras)
    baselines = []
    for k, (train_k, val_k, _) in enumerate(parts, 1):
        out = run_dir / "input" if n_folds == 1 else run_dir / "input" / f"fold_{k}"
        out.mkdir(parents=True, exist_ok=True)
        train_k.to_csv(out / "train.csv", index=False)
        for name, frame in extras.items():
            (out / "extra").mkdir(exist_ok=True)
            frame.to_csv(out / "extra" / f"{name}.csv", index=False)
        ids = hidden[(hidden["_fold"] == (1 if n_folds == 1 else k)) & (hidden["_split"] == "val")
                     | (hidden["_split"] == "test")]
        ids.sample(frac=1.0, random_state=spec.seed + k)[[ID_COL] + features].to_csv(
            out / "test.csv", index=False)
        baselines.append(_baseline(metric, train_k[spec.target], val_k[spec.target], positive_label))

    baseline = float(np.mean(baselines))
    if not np.isfinite(baseline):
        raise ValueError(f"The baseline {metric_name} score is not finite; the target may be constant "
                         "in a validation fold. Check the target column.")
    task = Task(
        spec=spec, run_dir=run_dir, kind=kind, metric_name=metric_name,
        positive_label=positive_label, classes=classes, n_train=len(parts[0][0]),
        profile=profile(profile_df, spec.target), baseline=baseline,
        n_folds=n_folds,
        extra_profile="\n\n".join(describe_table(Path(n), f, title=f"data/extra/{n}.csv") for n, f in extras.items()),
        docs_text=read_docs([Path(p) for p in spec.docs]),
    )
    task.save()
    return task


def _read_extras(paths: list[str]) -> dict[str, pd.DataFrame]:
    """Additional tables by the file name they get under data/extra/ (always plain CSV)."""
    out = {}
    for p in map(Path, paths):
        name = p.name.split(".")[0] or "table"
        while name in out:
            name += "_2"
        out[name] = read_table(p)
    return out


def _folds(df: pd.DataFrame, target: str, kind: str, n: int, seed: int):
    """k-fold indices; stratified for classification unless a class is too rare."""
    if kind == "classification":
        try:
            return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(df, df[target].astype(str)))
        except ValueError:
            pass
    return list(KFold(n, shuffle=True, random_state=seed).split(df))
