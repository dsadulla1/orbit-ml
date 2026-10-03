"""The custom model module: what `check_module` rejects and what the pickle scan allows."""
import pickle
import sys

import pytest

from orbit.pickle_scan import MAX_MODULE_BYTES, check_global, check_module, scan

BENIGN = '''
import math, re, json, datetime, itertools, collections, functools, typing, dataclasses
import statistics, string, decimal, fractions, warnings, numbers, operator, copy, enum, abc
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import lightgbm, xgboost
from orbit_helpers import Preprocessor

PATTERN = re.compile(r"\\d+")


class Ratio(BaseEstimator, TransformerMixin):
    def __init__(self, a="a", b="b"):
        self.a, self.b = a, b

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(X).copy()
        out["ratio"] = out[self.a] / (out[self.b].abs() + 1)
        out["n"] = out[self.a].astype(str).str.extract(r"(\\d+)", expand=False)
        out["w"] = getattr(out, "shape")[0]
        return out

    def __getstate__(self):
        return dict(self.__dict__)
'''

REJECTED = [
    ("import os", "'os'"),
    ("import sys", "'sys'"),
    ("import subprocess", "'subprocess'"),
    ("import socket", "'socket'"),
    ("import shutil", "'shutil'"),
    ("import pathlib", "'pathlib'"),
    ("import io", "'io'"),
    ("import requests", "'requests'"),
    ("import urllib.request", "'urllib.request'"),
    ("from http import client", "'http'"),
    ("import ctypes", "'ctypes'"),
    ("import importlib", "'importlib'"),
    ("import pickle", "'pickle'"),
    ("import marshal", "'marshal'"),
    ("import builtins", "'builtins'"),
    ("import tempfile", "'tempfile'"),
    ("import threading", "'threading'"),
    ("import multiprocessing", "'multiprocessing'"),
    ("import signal", "'signal'"),
    ("import os.path", "'os.path'"),
    ("import orbit_helpers_evil", "orbit_helpers_evil"),
    ("import pandas.io.parsers", "pandas.io"),
    ("from sklearn.datasets import load_iris", "sklearn.datasets"),
    ("from numpy import load", "'load'"),
    ("from pandas import read_csv", "read_csv"),
    ("from numpy import *", "import *"),
    ("from . import sibling", "relative"),
    ("from .sibling import x", "relative"),
    ("x = eval('1')", "'eval'"),
    ("exec('x = 1')", "'exec'"),
    ("f = compile('1', 'f', 'eval')", "'compile'"),
    ("f = open('data.csv')", "'open'"),
    ("m = __import__('os')", "__import__"),
    ("x = input()", "'input'"),
    ("breakpoint()", "'breakpoint'"),
    ("g = globals()", "'globals'"),
    ("g = locals()", "'locals'"),
    ("g = vars(object)", "'vars'"),
    ("e = eval", "'eval'"),
    ("getattr(obj, name)", "literal attribute name"),
    ("setattr(obj, name, 1)", "literal attribute name"),
    ("delattr(obj, 'a' + 'b')", "literal attribute name"),
    ("getattr(obj, '__class__')", "'__class__'"),
    ("setattr(obj, '__dict__', {})", "'__dict__'"),
    ("g = getattr", "may only be called"),
    ("f.__globals__", "__globals__"),
    ("__builtins__", "__builtins__"),
    ("object.__subclasses__()", "__subclasses__"),
    ("C.__bases__", "__bases__"),
    ("C.__mro__", "__mro__"),
    ("f.__code__", "__code__"),
    ("f.__closure__", "__closure__"),
    ("x = '{0.__globals__}'.format(f)", "__globals__"),
    ("self.__dict__ = {}", "__dict__"),
    ("self.__dict__['a'] = 1", "__dict__"),
    ("self.__dict__.update(a=1)", "__dict__"),
    ("class C:\n    def __reduce__(self):\n        return (print, ())", "__reduce__"),
    ("class C:\n    def __reduce_ex__(self, protocol):\n        return (print, ())", "__reduce_ex__"),
    ("class C:\n    __reduce__ = lambda self: (print, ())", "__reduce__"),
    ("import numpy as np\nnp.load('x.npy')", "'load'"),
    ("import pandas as pd\npd.read_csv('x.csv')", "'read_csv'"),
    ("import pandas as pd\npd.eval('1+1')", "'eval'"),
    ("import numpy as np\nnp.fromfile('x')", "'fromfile'"),
    ("import operator\noperator.attrgetter('x')", "'attrgetter'"),
    ("def f(:\n    pass", "not valid Python"),
]


def test_benign_module_with_every_allowed_import_passes():
    assert check_module(BENIGN) == []
    assert check_module("") == []


@pytest.mark.parametrize("source,needle", REJECTED, ids=[s.splitlines()[0][:40] for s, _ in REJECTED])
def test_each_forbidden_construct_is_rejected(source, needle):
    problems = check_module(source)
    assert problems and needle in " ".join(problems), problems


def test_oversized_module_is_rejected_without_parsing():
    source = "x = 1\n" * (MAX_MODULE_BYTES // 6 + 10)
    problems = check_module(source)
    assert len(problems) == 1 and "larger than" in problems[0]
    assert check_module("x = 1\n" * 1000) == []


def test_imports_inside_functions_and_classes_are_checked():
    src = "class A:\n    def f(self):\n        import os\n        return os.getcwd()\n"
    assert any("'os'" in p for p in check_module(src))


def test_violations_are_capped_and_name_the_line():
    problems = check_module("\n".join(f"import os{i}" for i in range(50)))
    assert len(problems) == 20 and problems[0].startswith("line 1:")


# ---- the pickle scan and the module name ----

def _stack_global(module: str, name: str) -> bytes:
    def short(text: str) -> bytes:
        raw = text.encode()
        return b"\x8c" + bytes([len(raw)]) + raw
    return b"\x80\x04" + short(module) + short(name) + b"\x93)R."


def test_scan_allows_orbit_custom_names_only_when_asked():
    data = _stack_global("orbit_custom", "Ratio")
    assert scan(data, custom=True) == []
    problems = scan(data)
    assert problems and "orbit_custom.py was not written" in problems[0]
    assert check_global("orbit_custom", "Ratio", custom=True) is None
    assert check_global("orbit_custom", "Outer.Inner", custom=True) is None


@pytest.mark.parametrize("module,name", [
    ("orbit_custom_evil", "X"), ("orbit_custom.sub", "X"), ("orbit_customs", "X"),
    ("my_orbit_custom", "X"), ("__main__", "X"), ("orbit_custom", "os.system"),
    ("orbit_custom", "__main__.X"), ("orbit_custom", "load_data"), ("orbit_custom", "eval_it"),
])
def test_scan_rejects_lookalike_modules_and_dangerous_names(module, name):
    assert scan(_stack_global(module, name), custom=True)


def test_a_real_pickle_of_a_custom_class_is_allowed(tmp_path, monkeypatch):
    simple = "\n".join(l for l in BENIGN.splitlines() if "lightgbm" not in l and "orbit_helpers" not in l)
    (tmp_path / "orbit_custom.py").write_text(simple)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("orbit_custom", None)
    import orbit_custom

    try:
        data = pickle.dumps(orbit_custom.Ratio(), protocol=4)
    finally:
        sys.modules.pop("orbit_custom", None)
    assert scan(data, custom=True) == [] and scan(data)
