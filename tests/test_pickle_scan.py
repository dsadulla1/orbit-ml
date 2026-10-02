import builtins
import os
import pickle
import pickletools
import subprocess

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from orbit.pickle_scan import check_global, scan

PROTOCOLS = [2, 4, 5]


def _frame(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                         "c": rng.choice(["x", "y", "z"], n)}), rng.integers(0, 2, n)


class Payload:
    """Pickles to a call of `func(*args)`; the scan must refuse the stream before anything runs."""

    def __init__(self, func, *args):
        self.func, self.args = func, args

    def __reduce__(self):
        return self.func, self.args


def _raw_global(module: str, name: str, protocol: int) -> bytes:
    """A stream that imports `module.name` and calls it, built by hand (no import is needed)."""
    if protocol >= 4:
        body = (b"\x8c" + bytes([len(module)]) + module.encode() + b"\x94"
                + b"\x8c" + bytes([len(name)]) + name.encode() + b"\x94\x93")
    else:
        body = b"c" + module.encode() + b"\n" + name.encode() + b"\n"
    return b"\x80" + bytes([protocol]) + body + b"N\x85R."


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_benign_pipelines_pass(protocol, orbit_helpers):
    X, y = _frame()
    cols = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]),
                                ["a", "b"]),
                               ("cat", OneHotEncoder(handle_unknown="ignore"), ["c"])])
    models = [
        Pipeline([("cols", cols), ("clf", LogisticRegression())]),
        Pipeline([("cols", cols), ("clf", RandomForestClassifier(n_estimators=5, random_state=0))]),
        Pipeline([("prep", orbit_helpers.Preprocessor()), ("clf", GradientBoostingClassifier(n_estimators=5))]),
    ]
    for model in models:
        model.fit(X, y)
        assert scan(pickle.dumps(model, protocol=protocol)) == []
    reg = Pipeline([("prep", orbit_helpers.Preprocessor(fill=False)),
                    ("reg", HistGradientBoostingRegressor(max_iter=5))]).fit(X, y)
    assert scan(pickle.dumps(reg, protocol=protocol)) == []


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_arrays_and_pandas_objects_pass(protocol):
    X, _ = _frame()
    # pandas 3 pickles string labels and columns through pyarrow, which the sandbox does not have
    X = X.drop(columns="c")
    X["when"] = pd.date_range("2020-01-01", periods=len(X))
    X["cat"] = pd.Categorical([1, 2, 3] * 20)
    X = X.set_axis(range(X.shape[1]), axis=1)
    objs = [np.arange(10), np.random.default_rng(1).normal(size=(4, 3)), np.array(["a", "b"]),
            np.datetime64("2020-01-01"), X, X.set_index(2), X[0], X.dtypes,
            {"k": [1, 2.5, "s", None, (1, 2), {3}, frozenset({4}), b"b", 3 + 4j]}]
    for obj in objs:
        assert scan(pickle.dumps(obj, protocol=protocol)) == [], obj


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("payload", [
    Payload(os.system, "echo hi"),
    Payload(builtins.eval, "1"),
    Payload(builtins.exec, "1"),
    Payload(subprocess.Popen, ["true"]),
    Payload(getattr, "abc", "upper"),
    Payload(np.load, "x.npy"),
    Payload(np.fromfile, "x.bin"),
    Payload(pd.read_pickle, "x.pkl"),
    Payload(open, "x"),
], ids=lambda p: getattr(p.func, "__qualname__", str(p.func)))
def test_payloads_are_rejected(payload, protocol):
    problems = scan(pickle.dumps(payload, protocol=protocol))
    assert problems and "is not allowed" in problems[0]


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_main_defined_function_is_rejected(protocol):
    problems = scan(_raw_global("__main__", "my_transform", protocol))
    assert "__main__.my_transform" in problems[0] and "never allowed" in problems[0]


@pytest.mark.parametrize("protocol", PROTOCOLS)
@pytest.mark.parametrize("module,name", [
    ("os", "system"), ("posix", "system"), ("nt", "system"), ("subprocess", "check_output"),
    ("sys", "exit"), ("socket", "socket"), ("shutil", "rmtree"), ("pickle", "loads"),
    ("builtins", "eval"), ("builtins", "getattr"), ("builtins", "__import__"), ("builtins", "open"),
    ("__builtin__", "eval"), ("numpy", "load"), ("numpy.lib.npyio", "NpzFile"),
    ("numpy.testing._private.utils", "runstring"), ("pandas", "read_csv"), ("pandas", "eval"),
    ("pandas.io.pickle", "read_pickle"), ("sklearn.datasets", "fetch_openml"),
    ("joblib", "load"), ("numpy", "os.system"), ("numpy", "memmap"), ("functools", "reduce"),
    ("_codecs", "open"), ("importlib", "import_module"), ("ctypes", "CDLL"),
    ("notalib", "thing"), ("orbit_helpers_evil", "x"),
])
def test_forbidden_globals_by_name(module, name, protocol):
    problems = scan(_raw_global(module, name, protocol))
    assert len(problems) == 1 and f"{module}.{name}" in problems[0]


def test_allowed_names_by_hand():
    for module, name in [("builtins", "set"), ("copyreg", "_reconstructor"), ("_codecs", "encode"),
                         ("functools", "partial"), ("collections", "OrderedDict"),
                         ("numpy", "ndarray"), ("sklearn.pipeline", "Pipeline"),
                         ("orbit_helpers", "Preprocessor"), ("lightgbm.sklearn", "LGBMClassifier"),
                         ("xgboost.sklearn", "XGBClassifier"), ("lightgbm.basic", "Booster")]:
        assert check_global(module, name) is None


def test_stack_global_with_memoized_names():
    # protocol 4 memoizes module and class names; a later STACK_GLOBAL re-fetches them with BINGET
    X, y = _frame()
    pipe = Pipeline([("a", StandardScaler()), ("b", StandardScaler()),
                     ("c", LogisticRegression())]).fit(X[["a", "b"]], y)
    blob = pickle.dumps(pipe, protocol=4)
    ops = [op.name for op, _, _ in pickletools.genops(blob)]
    assert "STACK_GLOBAL" in ops and ops.count("BINGET") + ops.count("LONG_BINGET") > 0
    assert scan(blob) == []
    # names pushed earlier, fetched from the memo: "os" at memo 0, "system" at memo 1
    evil = (b"\x80\x04\x8c\x02os\x94\x8c\x06system\x94" b"0" b"0"       # push both, pop both
            b"h\x00h\x01\x93\x8c\x04echo\x85R.")                      # re-fetch and call
    assert "os.system" in scan(evil)[0]
    # a name that is a memoized tuple is not a string: cannot be resolved
    unresolved = b"\x80\x04\x8c\x02os\x94\x8c\x06system\x94\x86\x94h\x02h\x00\x93."
    assert "cannot be resolved" in scan(unresolved)[0]


def test_stack_global_not_fooled_by_pops():
    # push os/system, push numpy/ndarray, drop the last two, then STACK_GLOBAL (takes os, system)
    stream = (b"\x80\x04\x8c\x02os\x8c\x06system\x8c\x05numpy\x8c\x07ndarray" b"00\x93N\x85R.")
    problems = scan(stream)
    assert problems and "cannot be resolved" in problems[0]


@pytest.mark.parametrize("data,expect", [
    (b"", "not a readable pickle"),
    (b"this is not a pickle at all", "not a readable pickle"),
    (pickle.dumps({"a": list(range(50))}, protocol=4)[:-5], "not a readable pickle"),
    (pickle.dumps([1, 2, 3])[:-1], "not a readable pickle"),
    (pickle.dumps([1, 2, 3]) + b"extra bytes", "bytes follow the end"),
    (pickle.dumps([1]) + pickle.dumps(Payload(os.system, "x")), "bytes follow the end"),
])
def test_unreadable_and_trailing_data(data, expect):
    problems = scan(data)
    assert len(problems) == 1 and expect in problems[0]


def test_persistent_ids_and_extension_codes_rejected():
    assert "persistent ids" in scan(b"\x80\x04N\x51.")[0]          # BINPERSID
    assert "extension" in scan(b"\x80\x04\x82\x01.")[0]            # EXT1


def test_bytes_input_is_not_mutated_and_nothing_runs(tmp_path):
    marker = tmp_path / "ran"
    blob = pickle.dumps(Payload(marker.write_text, "x"), protocol=4)
    assert scan(blob)
    assert not marker.exists()
    assert scan(bytearray(pickle.dumps([1]))) == []


def test_violations_are_capped_and_deduplicated():
    blob = b"\x80\x02" + b"".join(b"cos\nsystem\n" for _ in range(5)) + b"."
    assert len(scan(blob)) == 1
    many = b"\x80\x02" + b"".join(f"cos\nfunc{i}\n".encode() for i in range(100)) + b"."
    assert len(scan(many)) <= 20
