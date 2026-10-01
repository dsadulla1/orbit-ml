"""Resolve user-supplied data sources (local files, directories, URLs, zip archives).

Everything here runs on the orchestrator, never in the sandbox. Sources are copied into the run
directory so a run stays self-contained. Tables (CSV/TSV, optionally gzipped) and short text
documents (data descriptions, READMEs) are collected; anything else is skipped with a note.

URLs are fetched with the host's network access, so only point orbit at sources you trust.
"""
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

TABLE_SUFFIXES = (".csv", ".tsv", ".csv.gz", ".tsv.gz")
DOC_SUFFIXES = (".txt", ".md", ".rst")
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024
MAX_UNZIPPED_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILES = 200
MAX_DEPTH = 3
# Folders never copied from a source directory: earlier orbit runs (their hidden labels must not
# become an input table), tracker output and tooling.
SKIP_DIRS = {"runs", "hidden", "mlruns", "wandb", "__pycache__", "node_modules", "venv"}
_CONTENT_TYPES = {"text/csv": ".csv", "text/tab-separated-values": ".tsv", "text/plain": ".txt",
                  "application/zip": ".zip", "application/gzip": ".csv.gz"}


class SourceError(ValueError):
    """A data source could not be fetched or understood; the message is shown to the user."""


@dataclass
class Resolved:
    tables: list[Path] = field(default_factory=list)
    docs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


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
    return None


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
                                  "expected CSV, TSV, gzip, zip or a text document")
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


def _collect(path: Path, dest: Path, out: Resolved, depth: int = 0) -> None:
    if path.is_dir():
        if depth > MAX_DEPTH:
            out.notes.append(f"skipped {path.name}/: nested more than {MAX_DEPTH} levels deep")
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
    elif kind == "doc":
        out.docs.append(path)
    elif not path.name.startswith("."):
        out.notes.append(f"skipped {path.name}: unsupported file type")


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
            _copy_tree(path, copied, avoid=dest)
            _collect(copied, home, out)
        else:
            if _kind(path) is None:
                raise SourceError(f"{source!r}: unsupported file type; expected CSV, TSV, gzip, "
                                  "zip or a text document")
            shutil.copy2(path, home / path.name)
            _collect(home / path.name, home, out)
    if len(out.tables) + len(out.docs) > MAX_FILES:
        raise SourceError(f"more than {MAX_FILES} files found; point orbit at a smaller folder")
    if not out.tables:
        raise SourceError("no CSV/TSV table found in the given sources")
    return out


def _copy_tree(src: Path, dst: Path, avoid: Path | None = None) -> None:
    """Copy a directory without following symlinks. Skips hidden and tool folders, orbit run
    folders, the destination (`avoid`), and stops at the file-count and byte limits."""
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
        if p.is_file() and _kind(p) is not None:
            count += 1
            total += p.stat().st_size
            if count > MAX_FILES:
                raise SourceError(f"more than {MAX_FILES} files in {src}; point orbit at a smaller folder")
            if total > MAX_UNZIPPED_BYTES:
                raise SourceError(f"{src} holds more than {MAX_UNZIPPED_BYTES / 1e9:.0f} GB of data files")
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst / rel)
