"""Resolve user-supplied data sources (local files, directories, URLs, zip archives).

Everything here runs on the orchestrator, never in the sandbox. Sources are copied into the run
directory so a run stays self-contained. Tables (CSV/TSV, optionally gzipped, and Excel .xlsx/.xlsm
workbooks, whose sheets are converted to CSV next to the workbook) and short text documents (data
descriptions, READMEs) are collected; anything else is skipped with a note.

URLs are fetched with the host's network access, so only point orbit at sources you trust.
"""
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TABLE_SUFFIXES = (".csv", ".tsv", ".csv.gz", ".tsv.gz")
DOC_SUFFIXES = (".txt", ".md", ".rst")
EXCEL_SUFFIXES = (".xlsx", ".xlsm")
SUPPORTED = "CSV, TSV, gzip, zip, Excel .xlsx, or .txt/.md/.rst documents"
MAX_NOTES = 10  # skipped-file notes kept per run; the rest are only counted
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
MAX_UNZIPPED_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILES = 200
MAX_DEPTH = 3
# Folders never copied from a source directory: earlier orbit runs (their hidden labels must not
# become an input table), tracker output and tooling.
SKIP_DIRS = {"runs", "hidden", "mlruns", "wandb", "__pycache__", "node_modules", "venv"}
_CONTENT_TYPES = {"text/csv": ".csv", "text/tab-separated-values": ".tsv", "text/plain": ".txt",
                  "application/zip": ".zip", "application/gzip": ".csv.gz",
                  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                  "application/vnd.ms-excel": ".xls"}
_XLS_NOTE = "old .xls format is not supported; save it as .xlsx or CSV"


class SourceError(ValueError):
    """A data source could not be fetched or understood; the message is shown to the user."""


@dataclass
class Resolved:
    tables: list[Path] = field(default_factory=list)
    docs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: int = 0  # files skipped so far (only the first few get a note)
    converted_bytes: int = 0  # CSV written from Excel sheets


def is_url(source: str) -> bool:
    return urllib.parse.urlparse(source).scheme in ("http", "https")


def _kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("."):
        return None
    if name.endswith(TABLE_SUFFIXES):
        return "table"
    if name.endswith(DOC_SUFFIXES):
        return "doc"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(EXCEL_SUFFIXES):
        return "xlsx"
    if name.endswith(".xls"):
        return "xls"
    return None


def _skip(out: Resolved, name: str, why: str) -> None:
    if out.skipped < MAX_NOTES:
        out.notes.append(f"skipped {name}: {why}")
    out.skipped += 1


class _HttpOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme not in ("http", "https"):
            raise SourceError(f"refusing redirect to {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, dest_dir: Path, max_bytes: int = MAX_DOWNLOAD_BYTES, timeout: int = 30) -> Path:
    opener = urllib.request.build_opener(_HttpOnly)
    try:
        resp = opener.open(urllib.request.Request(url, headers={"User-Agent": "orbit"}), timeout=timeout)
    except SourceError:
        raise
    except Exception as e:  # noqa: BLE001 - network, DNS, HTTP status: all reported the same way
        raise SourceError(f"could not download {url}: {e}") from e
    with resp:
        raw = urllib.parse.unquote(urllib.parse.urlparse(resp.geturl()).path).replace("\x00", "")
        name = Path(raw).name or "download"
        if _kind(Path(name)) is None:
            ctype = resp.headers.get_content_type()
            if ctype not in _CONTENT_TYPES:
                raise SourceError(f"{url}: unsupported file type {name!r} ({ctype}); "
                                  f"expected {SUPPORTED}")
            name += _CONTENT_TYPES[ctype]
        length = resp.headers.get("Content-Length", "")
        if length.isdigit() and int(length) > max_bytes:
            raise SourceError(f"{url} is {int(length) / 1e6:.0f} MB; the limit is {max_bytes / 1e6:.0f} MB")
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / name
        size = 0
        with open(path, "wb") as f:
            while chunk := resp.read(1 << 20):
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    path.unlink()
                    raise SourceError(f"{url} is larger than the {max_bytes / 1e6:.0f} MB limit")
                f.write(chunk)
    return path


def _unzip(archive: Path, dest: Path, max_bytes: int = MAX_UNZIPPED_BYTES) -> Path:
    """Extract only regular files whose names stay inside `dest`; refuse oversized archives."""
    out = dest / archive.stem
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > max_bytes:
            raise SourceError(f"{archive.name}: too many or too large files to extract")
        for info in infos:
            target = (out / info.filename).resolve()
            if not target.is_relative_to(out.resolve()):
                raise SourceError(f"{archive.name}: unsafe path {info.filename!r} in archive")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:  # symlink entry
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return out


def _convert_workbook(path: Path, out: Resolved) -> list[Path]:
    """Write every sheet with data to a CSV next to the workbook and return those paths.

    Sheets are read one at a time; dates are written by pandas as ISO strings (the time of day is
    kept unless every value is midnight). Unreadable workbooks raise SourceError."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        if magic != b"PK\x03\x04":
            raise SourceError(f"{path.name}: not a readable .xlsx workbook (it may be password-protected, "
                              "an old .xls file renamed, or corrupt)")
        with zipfile.ZipFile(path) as z:  # a workbook is a zip: refuse decompression bombs
            if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED_BYTES:
                raise SourceError(f"{path.name}: the workbook expands to more than "
                                  f"{MAX_UNZIPPED_BYTES / 1e9:.0f} GB")
        tables: list[Path] = []
        with pd.ExcelFile(path, engine="openpyxl") as book:
            names = [str(n) for n in book.sheet_names]
            for sheet in names:
                df = book.parse(sheet)
                if df.empty or df.shape[1] == 0:
                    _skip(out, f"{path.name} [{sheet}]", "sheet has no data")
                    continue
                label = re.sub(r"[^A-Za-z0-9_-]+", "_", sheet).strip("_") or "sheet"
                stem = path.stem if len(names) == 1 else f"{path.stem}__{label}"
                csv, n = path.with_name(stem + ".csv"), 2
                while csv.exists():
                    csv, n = path.with_name(f"{stem}_{n}.csv"), n + 1
                df.to_csv(csv, index=False)
                out.converted_bytes += csv.stat().st_size
                if out.converted_bytes > MAX_UNZIPPED_BYTES or len(out.tables) + len(tables) >= MAX_FILES:
                    raise SourceError(f"{path.name}: too many or too large sheets to convert")
                tables.append(csv)
                del df
    except SourceError:
        raise
    except Exception as e:  # noqa: BLE001 - corrupt zip, bad XML, missing parts, encryption
        raise SourceError(f"{path.name}: could not read the Excel workbook "
                          f"({type(e).__name__}: {str(e)[:120]})") from e
    return tables


def _collect(path: Path, dest: Path, out: Resolved, depth: int = 0) -> None:
    if path.is_dir():
        if depth > MAX_DEPTH:
            _skip(out, f"{path.name}/", f"nested more than {MAX_DEPTH} levels deep")
            return
        for child in sorted(path.iterdir()):
            if not child.name.startswith("."):
                _collect(child, dest, out, depth + 1)
        return
    kind = _kind(path)
    if kind == "zip":
        _collect(_unzip(path, dest), dest, out, depth + 1)
    elif kind == "table":
        out.tables.append(path)
    elif kind == "xlsx":
        out.tables.extend(_convert_workbook(path, out))
    elif kind == "xls":
        _skip(out, path.name, _XLS_NOTE)
    elif kind == "doc":
        out.docs.append(path)
    elif not path.name.startswith("."):
        _skip(out, path.name, "unsupported file type")


def resolve(sources: list[str], dest: Path) -> Resolved:
    """Fetch/copy `sources` under `dest` and return the tables and documents found."""
    if not sources:
        raise SourceError("no data source given")
    dest.mkdir(parents=True, exist_ok=True)
    out = Resolved()
    for i, source in enumerate(sources):
        home = dest / f"src{i + 1}"
        if is_url(source):
            _collect(download(source, home), home, out)
            continue
        path = Path(source).expanduser()
        if not path.exists():
            raise SourceError(f"{source!r} does not exist (and is not an http(s) URL)")
        home.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            copied = home / path.name
            _copy_tree(path, copied, avoid=dest, out=out)
            _collect(copied, home, out)
        else:
            kind = _kind(path)
            if kind == "xls":
                raise SourceError(f"{source!r}: {_XLS_NOTE}")
            if kind is None:
                raise SourceError(f"{source!r}: unsupported file type; expected {SUPPORTED}")
            shutil.copy2(path, home / path.name)
            _collect(home / path.name, home, out)
    if len(out.tables) + len(out.docs) > MAX_FILES:
        raise SourceError(f"more than {MAX_FILES} files found; point orbit at a smaller folder")
    if out.skipped > MAX_NOTES:
        out.notes.append(f"... and {out.skipped - MAX_NOTES} more files skipped")
    if not out.tables:
        found = "; ".join(n.removeprefix("skipped ") for n in out.notes[:MAX_NOTES])
        raise SourceError("no CSV/TSV or Excel table found in the given sources"
                          + (f" (skipped: {found})" if found else "")
                          + (f"; found {len(out.docs)} text document(s) but no data" if out.docs else "")
                          + f". Supported: {SUPPORTED}.")
    return out


def _copy_tree(src: Path, dst: Path, avoid: Path | None = None, out: Resolved | None = None) -> None:
    """Copy a directory without following symlinks. Skips hidden and tool folders, orbit run
    folders, the destination (`avoid`), and stops at the file-count and byte limits. Files of
    unsupported types are not copied; they are noted on `out`."""
    avoid = avoid.resolve() if avoid else None
    # Folders that are orbit runs (task.json next to hidden/), wherever they were saved.
    runs = [t.parent for t in src.rglob("task.json") if (t.parent / "hidden").is_dir()]
    count = total = 0
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if p.is_symlink() or any(part.startswith(".") or part in SKIP_DIRS for part in rel.parts):
            continue
        if (avoid and p.resolve().is_relative_to(avoid)) or any(r in p.parents for r in runs):
            continue
        if len(rel.parts) > MAX_DEPTH + 1 and p.is_file():
            continue
        if p.is_file() and _kind(p) is None and out is not None:
            _skip(out, p.name, "unsupported file type")
        if p.is_file() and _kind(p) is not None:
            count += 1
            total += p.stat().st_size
            if count > MAX_FILES:
                raise SourceError(f"more than {MAX_FILES} files in {src}; point orbit at a smaller folder")
            if total > MAX_UNZIPPED_BYTES:
                raise SourceError(f"{src} holds more than {MAX_UNZIPPED_BYTES / 1e9:.0f} GB of data files")
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
