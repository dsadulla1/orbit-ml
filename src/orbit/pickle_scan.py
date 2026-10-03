"""Static allow-list scan of a pickle stream, run before an exported model is ever loaded.

`scan(data)` walks the opcodes with `pickletools.genops` (nothing is unpickled), resolves every
global the stream would import (GLOBAL, INST and STACK_GLOBAL, including names fetched back from the
memo) and returns a list of readable violations; an empty list means the stream only references
allowed callables. Streams that cannot be parsed, have bytes after the final STOP (a `joblib.dump`
file), use extension codes or persistent ids, or build a STACK_GLOBAL name in a way the scan cannot
follow are rejected too.

Rules, in order:
  * top-level modules `__main__`, `os`, `posix`, `nt`, `sys`, `subprocess`, `socket`, `shutil`,
    `pickle`, ... are always rejected, also as a part of a dotted attribute name;
  * the module must be one of numpy, pandas, scipy, sklearn, lightgbm, xgboost, joblib,
    orbit_helpers, collections, datetime, decimal, fractions, or a name from the short lists
    below (`builtins`: plain constructors such as list, dict, set, int; `copyreg`, `_codecs`,
    `functools`: only what pickling itself needs);
  * inside the allowed libraries, names that read files, run code or reach the network are
    rejected: any module or attribute part containing load, dump, fromfile, system, popen, eval,
    exec, runstring, subprocess, open, memmap, compile, import, ctypes, testing or fetch, or
    starting with read, to_, save or write; and the modules numpy.lib.npyio, pandas.io,
    sklearn.datasets and scipy.io.

A model may carry its own classes and functions in one module file, `orbit_custom.py`, written next
to `model.pkl`. With `scan(data, custom=True)` the top-level module name `orbit_custom` (exactly:
not `orbit_custom_x`, not `orbit_custom.sub`) is allowed too, and the deny-list applies to the names
inside it. `check_module(source)` checks that file's source with `ast` before it is accepted.

This is defence in depth, not a security guarantee: an allow-listed library may still contain a
callable that does harm, a pickle can run code in `__setstate__` of an allowed class, and a static
check of a Python module cannot rule out every way around it. The real boundary is that the model is
only ever loaded inside a dedicated container without credentials or network access.
"""
import ast
import io
import pickletools

ALLOWED_MODULES = ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "joblib",
                   "orbit_helpers", "collections", "datetime", "decimal", "fractions")
ALLOWED_NAMES = {  # modules where only these names may be used
    "builtins": {"object", "set", "frozenset", "list", "dict", "tuple", "slice", "range",
                 "complex", "bytearray", "bytes", "int", "float", "str", "bool"},
    "copyreg": {"_reconstructor", "__newobj__", "__newobj_ex__"},
    "_codecs": {"encode"},
    "functools": {"partial"},
}
ALLOWED_NAMES["__builtin__"] = ALLOWED_NAMES["builtins"]
ALWAYS_REJECTED = {"__main__", "os", "posix", "nt", "sys", "subprocess", "socket", "shutil",
                   "pickle", "_pickle", "importlib", "ctypes", "io", "pathlib", "runpy", "code",
                   "marshal", "types", "commands", "popen2", "pty", "signal", "threading",
                   "multiprocessing", "urllib", "http", "ftplib", "tempfile", "webbrowser"}
DENIED_PARTS = ("load", "dump", "fromfile", "system", "popen", "eval", "exec", "runstring",
                "subprocess", "open", "memmap", "compile", "import", "ctypes", "testing", "fetch")
DENIED_PREFIXES = ("read", "to_", "save", "write")
DENIED_MODULES = ("numpy.lib.npyio", "numpy.lib._npyio_impl", "pandas.io", "sklearn.datasets",
                  "scipy.io")
MAX_VIOLATIONS = 20

CUSTOM_MODULE = "orbit_custom"
CUSTOM_FILE = "orbit_custom.py"
MAX_MODULE_BYTES = 200 * 1024
MODULE_IMPORTS = ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "orbit_helpers",
                  "math", "re", "json", "datetime", "itertools", "collections", "functools", "typing",
                  "dataclasses", "statistics", "string", "decimal", "fractions", "warnings", "numbers",
                  "operator", "copy", "enum", "abc")
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals",
                   "locals", "vars"}
DYNAMIC_ATTR_CALLS = {"getattr", "setattr", "delattr"}
FORBIDDEN_DUNDERS = {"__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__",
                     "__code__", "__closure__", "__getattribute__", "__loader__", "__spec__",
                     "__import__"}
FORBIDDEN_DEFS = {"__reduce__", "__reduce_ex__"}
# attributes that read or write files, run code or load libraries; `re.compile` and `json.loads`
# are fine, so only names that are unambiguous are listed
FORBIDDEN_ATTRS = {"eval", "exec", "open", "popen", "system", "load", "dump", "save", "savez",
                   "savez_compressed", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap",
                   "testing", "ctypeslib", "f2py", "distutils", "attrgetter", "methodcaller"}
FORBIDDEN_ATTR_PREFIXES = ("read_", "fetch_")

_STRING_OPS = {"STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE",
               "SHORT_BINUNICODE", "BINUNICODE8"}
_GET_OPS = {"GET", "BINGET", "LONG_BINGET"}
_PUT_OPS = {"PUT", "BINPUT", "LONG_BINPUT"}
_NEUTRAL_OPS = {"PROTO", "FRAME", "NOP"}


def _in_modules(module: str, prefixes) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def check_global(module: str, name: str, custom: bool = False) -> str | None:
    """Why `module.name` is not allowed, or None if it is. `custom` allows the module
    `orbit_custom` (only when its file travels with the model)."""
    parts = module.split(".") + name.split(".")
    bad = next((p for p in [parts[0]] + name.split(".") if p in ALWAYS_REJECTED), None)
    if bad:
        return f"module or name {bad!r} is never allowed"
    if module == CUSTOM_MODULE and not custom:
        return f"{CUSTOM_FILE} was not written next to model.pkl"
    if module in ALLOWED_NAMES:
        if name not in ALLOWED_NAMES[module]:
            return f"{module} is only allowed for {sorted(ALLOWED_NAMES[module])}"
        return None
    if module != CUSTOM_MODULE:
        if not _in_modules(module, ALLOWED_MODULES):
            return "module is not on the allow-list"
        if _in_modules(module, DENIED_MODULES):
            return "module can read files or run code"
    else:
        parts = name.split(".")
    for p in map(str.lower, parts):
        hit = next((d for d in DENIED_PARTS if d in p), None) or next(
            (d for d in DENIED_PREFIXES if p.startswith(d)), None)
        if hit:
            return f"{p!r} matches the deny-list entry {hit!r}"
    return None


def scan(data: bytes, custom: bool = False) -> list[str]:
    """Violations found in the pickle stream `data` (empty: allowed). Never unpickles. `custom`
    allows globals of the module `orbit_custom` (see the module docstring)."""
    found: list[str] = []

    def report(msg: str) -> None:
        if msg not in found and len(found) < MAX_VIOLATIONS:
            found.append(msg)

    def check(module: str, name: str) -> None:
        why = check_global(module, name, custom)
        if why:
            report(f"global {module}.{name} is not allowed: {why}")

    stream = io.BytesIO(bytes(data))
    memo: dict[int, str | None] = {}
    tail: list[str | None] = []   # strings pushed by consecutive pushes; reset by any other opcode
    last: str | None = None       # the top of the stack, when it is a known string
    try:
        for op, arg, _ in pickletools.genops(stream):
            n = op.name
            if n in _NEUTRAL_OPS:
                continue
            if n in _PUT_OPS or n == "MEMOIZE":
                memo[len(memo) if n == "MEMOIZE" else arg] = last
                continue
            if n in _STRING_OPS:
                last = arg
                tail = (tail + [arg])[-2:]
                continue
            if n in _GET_OPS:
                last = memo.get(arg)
                tail = (tail + [last])[-2:]
                continue
            if n in ("GLOBAL", "INST"):
                module, _, name = str(arg).partition(" ")
                check(module, name)
            elif n == "STACK_GLOBAL":
                if len(tail) == 2 and all(isinstance(t, str) for t in tail):
                    check(*tail)
                else:
                    report("STACK_GLOBAL with a module or name that cannot be resolved statically")
            elif n in ("EXT1", "EXT2", "EXT4"):
                report("extension-registry globals are not allowed")
            elif n in ("PERSID", "BINPERSID"):
                report("persistent ids are not allowed")
            last, tail = None, []
    except Exception as e:  # noqa: BLE001 - genops raises ValueError, EOFError, ... on bad input
        found.append(f"not a readable pickle stream: {type(e).__name__}: {e}")
        return found
    if stream.tell() != len(stream.getbuffer()):
        report(f"{len(stream.getbuffer()) - stream.tell()} bytes follow the end of the pickle "
               "(not a plain `pickle.dump` file; joblib files are not supported)")
    return found


def _dotted_denied(module: str) -> bool:
    parts = module.split(".")
    return _in_modules(module, DENIED_MODULES) or any(
        p in FORBIDDEN_ATTRS or p.startswith(FORBIDDEN_ATTR_PREFIXES) for p in parts[1:])


def check_module(source: str) -> list[str]:
    """Violations in the source of a custom model module (empty: acceptable). Nothing is imported
    or run: the source is parsed with `ast` and checked for imports outside `MODULE_IMPORTS`, star
    and relative imports, calls to eval, exec, compile, open, `__import__`, input, breakpoint,
    globals, locals and vars, getattr/setattr/delattr with a computed or dunder name, access to
    `__globals__`, `__builtins__`, `__subclasses__`, `__mro__`, `__code__` and similar, writes to
    `__dict__`, definitions of `__reduce__`/`__reduce_ex__`, and attributes that read or write files
    (`read_csv`, `load`, `to_pickle`...). Like `scan`, this is defence in depth: the model still only
    runs in the sandbox and the serving container."""
    if len(source.encode("utf-8", "replace")) > MAX_MODULE_BYTES:
        return [f"{CUSTOM_FILE} is larger than {MAX_MODULE_BYTES // 1024} KB"]
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as e:
        return [f"{CUSTOM_FILE} is not valid Python: {type(e).__name__}: {e}"]
    found: list[str] = []

    def report(node, msg: str) -> None:
        text = f"line {getattr(node, 'lineno', 0)}: {msg}"
        if text not in found and len(found) < MAX_VIOLATIONS:
            found.append(text)

    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if not _in_modules(a.name, MODULE_IMPORTS) or _dotted_denied(a.name):
                    report(node, f"import of {a.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                report(node, "relative imports are not allowed")
            elif not _in_modules(mod, MODULE_IMPORTS) or _dotted_denied(mod):
                report(node, f"import from {mod!r} is not allowed")
            else:
                for a in node.names:
                    if a.name == "*":
                        report(node, "`import *` is not allowed")
                    elif _dotted_denied(f"{mod}.{a.name}") or a.name in FORBIDDEN_ATTRS or \
                            a.name.startswith(FORBIDDEN_ATTR_PREFIXES):
                        report(node, f"importing {a.name!r} from {mod!r} is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALLS or node.id in FORBIDDEN_DUNDERS:
                report(node, f"{node.id!r} is not allowed")
            elif node.id in DYNAMIC_ATTR_CALLS and id(node) not in called:
                report(node, f"{node.id!r} may only be called")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_DUNDERS:
                report(node, f"attribute {node.attr!r} is not allowed")
            elif node.attr in FORBIDDEN_ATTRS or node.attr.startswith(FORBIDDEN_ATTR_PREFIXES):
                report(node, f"attribute {node.attr!r} is not allowed (file, code or library access)")
            if node.attr == "__dict__" and not isinstance(node.ctx, ast.Load):
                report(node, "writing to __dict__ is not allowed")
        elif isinstance(node, ast.Subscript):
            v = node.value
            if (isinstance(v, ast.Attribute) and v.attr == "__dict__"
                    and not isinstance(node.ctx, ast.Load)):
                report(node, "writing to __dict__ is not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in FORBIDDEN_DEFS:
                report(node, f"defining {node.name} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            hit = next((d for d in FORBIDDEN_DUNDERS if d in node.value), None)
            if hit:
                report(node, f"the string mentions {hit!r}")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in DYNAMIC_ATTR_CALLS:
                arg = node.args[1] if len(node.args) > 1 else None
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                    report(node, f"{f.id} needs a literal attribute name")
                elif arg.value.startswith("__") or arg.value in FORBIDDEN_ATTRS:
                    report(node, f"{f.id} of {arg.value!r} is not allowed")
            if (isinstance(f, ast.Attribute) and f.attr in ("update", "pop", "popitem", "clear",
                                                              "setdefault")
                    and isinstance(f.value, ast.Attribute) and f.value.attr == "__dict__"):
                report(node, "writing to __dict__ is not allowed")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in FORBIDDEN_DEFS:
                    report(node, f"defining {t.id} is not allowed")
    return found
