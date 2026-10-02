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

This is defence in depth, not a security guarantee: an allow-listed library may still contain a
callable that does harm, and a pickle can run code in `__setstate__` of an allowed class. The real
boundary is that the model is only ever loaded inside a dedicated container without credentials or
network access.
"""
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

_STRING_OPS = {"STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE",
               "SHORT_BINUNICODE", "BINUNICODE8"}
_GET_OPS = {"GET", "BINGET", "LONG_BINGET"}
_PUT_OPS = {"PUT", "BINPUT", "LONG_BINPUT"}
_NEUTRAL_OPS = {"PROTO", "FRAME", "NOP"}


def _in_modules(module: str, prefixes) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def check_global(module: str, name: str) -> str | None:
    """Why `module.name` is not allowed, or None if it is."""
    parts = module.split(".") + name.split(".")
    bad = next((p for p in [parts[0]] + name.split(".") if p in ALWAYS_REJECTED), None)
    if bad:
        return f"module or name {bad!r} is never allowed"
    if module in ALLOWED_NAMES:
        if name not in ALLOWED_NAMES[module]:
            return f"{module} is only allowed for {sorted(ALLOWED_NAMES[module])}"
        return None
    if not _in_modules(module, ALLOWED_MODULES):
        return "module is not on the allow-list"
    if _in_modules(module, DENIED_MODULES):
        return "module can read files or run code"
    for p in map(str.lower, parts):
        hit = next((d for d in DENIED_PARTS if d in p), None) or next(
            (d for d in DENIED_PREFIXES if p.startswith(d)), None)
        if hit:
            return f"{p!r} matches the deny-list entry {hit!r}"
    return None


def scan(data: bytes) -> list[str]:
    """Violations found in the pickle stream `data` (empty: allowed). Never unpickles."""
    found: list[str] = []

    def report(msg: str) -> None:
        if msg not in found and len(found) < MAX_VIOLATIONS:
            found.append(msg)

    def check(module: str, name: str) -> None:
        why = check_global(module, name)
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
