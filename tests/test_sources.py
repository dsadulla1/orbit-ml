import functools
import http.server
import threading
import zipfile

import pandas as pd
import pytest

from orbit import sources
from orbit.sources import SourceError, resolve


def _write(path, text="a,b\n1,2\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_local_file(tmp_path):
    csv = _write(tmp_path / "in" / "train.csv")
    r = resolve([str(csv)], tmp_path / "run" / "source")
    assert [p.name for p in r.tables] == ["train.csv"] and r.docs == []
    assert r.tables[0].read_text() == csv.read_text() and tmp_path / "run" in r.tables[0].parents


def test_directory_collects_tables_docs_and_skips_junk(tmp_path):
    d = tmp_path / "house"
    _write(d / "train.csv"); _write(d / "extra" / "weather.tsv"); _write(d / "data_description.txt", "docs")
    _write(d / "model.pkl", "x"); _write(d / ".hidden.csv"); (d / "link.csv").symlink_to(d / "train.csv")
    r = resolve([str(d)], tmp_path / "src")
    assert sorted(p.name for p in r.tables) == ["train.csv", "weather.tsv"]
    assert [p.name for p in r.docs] == ["data_description.txt"]


def test_zip_extracts_and_rejects_path_traversal(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("data/train.csv", "a,b\n1,2\n")
        z.writestr("README.md", "hello")
    r = resolve([str(good)], tmp_path / "src")
    assert [p.name for p in r.tables] == ["train.csv"] and [p.name for p in r.docs] == ["README.md"]
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../../evil.csv", "a\n1\n")
    with pytest.raises(SourceError, match="unsafe path"):
        resolve([str(bad)], tmp_path / "src2")
    assert not (tmp_path / "evil.csv").exists()


def test_errors_are_readable(tmp_path):
    with pytest.raises(SourceError, match="does not exist"):
        resolve([str(tmp_path / "missing.csv")], tmp_path / "a")
    odd = _write(tmp_path / "model.pkl", "x")
    with pytest.raises(SourceError, match="unsupported file type"):
        resolve([str(odd)], tmp_path / "b")
    docs_only = tmp_path / "docs"
    _write(docs_only / "notes.txt", "x")
    with pytest.raises(SourceError, match="no CSV/TSV"):
        resolve([str(docs_only)], tmp_path / "c")
    with pytest.raises(SourceError, match="no data source"):
        resolve([], tmp_path / "d")


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "www"
    _write(root / "train.csv", "a,b\n1,2\n3,4\n")
    _write(root / "blob.bin", "x")
    with zipfile.ZipFile(root / "bundle.zip", "w") as z:
        z.writestr("t.csv", "a\n1\n")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "file:///etc/passwd")
                self.end_headers()
            else:
                super().do_GET()

        def log_message(self, *a):
            pass

    handler = functools.partial(Handler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_url_download_and_zip(server, tmp_path):
    r = resolve([f"{server}/train.csv", f"{server}/bundle.zip"], tmp_path / "src")
    assert sorted(p.name for p in r.tables) == ["t.csv", "train.csv"]


def test_url_errors(server, tmp_path, monkeypatch):
    with pytest.raises(SourceError, match="unsupported file type"):
        resolve([f"{server}/blob.bin"], tmp_path / "a")
    with pytest.raises(SourceError, match="could not download"):
        resolve([f"{server}/nope.csv"], tmp_path / "b")
    with pytest.raises(SourceError, match="refusing redirect|could not download"):
        resolve([f"{server}/redirect"], tmp_path / "c")
    monkeypatch.setattr(sources, "MAX_DOWNLOAD_BYTES", 4)
    with pytest.raises(SourceError, match="limit"):
        sources.download(f"{server}/train.csv", tmp_path / "d", max_bytes=4)
    assert not list((tmp_path / "d").glob("*.csv"))


def test_directory_copy_skips_earlier_runs_and_the_output_folder(tmp_path):
    proj = tmp_path / "proj"
    _write(proj / "data.csv")
    _write(proj / "runs" / "old" / "hidden" / "labels.csv")
    _write(proj / "results" / "old" / "task.json", "{}")                 # an earlier run in a custom --out
    _write(proj / "results" / "old" / "input" / "train.csv")
    (proj / "results" / "old" / "hidden").mkdir()
    _write(proj / "hidden" / "labels.csv")
    run_dir = proj / "results" / "run1"
    r = resolve([str(proj)], run_dir / "source")
    assert [p.name for p in r.tables] == ["data.csv"]


def _book(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    return path


def test_xlsx_single_sheet_keeps_datetimes(tmp_path):
    df = pd.DataFrame({"when": pd.to_datetime(["2010-12-01 08:26:00", "2011-01-02 17:05:30"]), "n": [1, 2]})
    r = resolve([str(_book(tmp_path / "in" / "Online Retail.xlsx", {"Sheet1": df}))], tmp_path / "src")
    assert [p.name for p in r.tables] == ["Online Retail.csv"]
    text = r.tables[0].read_text()
    assert "2010-12-01 08:26:00" in text and "2011-01-02 17:05:30" in text


def test_xlsx_several_sheets_skips_empty_and_sanitizes(tmp_path):
    book = _book(tmp_path / "w.xlsx", {"Train 1!": pd.DataFrame({"a": [1]}), "Empty": pd.DataFrame(),
                                      "test": pd.DataFrame({"a": [2]})})
    r = resolve([str(book)], tmp_path / "src")
    assert sorted(p.name for p in r.tables) == ["w__Train_1.csv", "w__test.csv"]
    assert any("Empty" in n and "no data" in n for n in r.notes)


def test_xlsx_inside_zip_and_folder(tmp_path):
    book = _book(tmp_path / "b" / "d.xlsx", {"S": pd.DataFrame({"a": [1, 2]})})
    archive = tmp_path / "d.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(book, "inner/d.xlsx")
    assert [p.name for p in resolve([str(archive)], tmp_path / "s1").tables] == ["d.csv"]
    assert [p.name for p in resolve([str(book.parent)], tmp_path / "s2").tables] == ["d.csv"]


def test_xlsx_served_by_url(tmp_path):
    book = _book(tmp_path / "web" / "data", {"S": pd.DataFrame({"a": [1, 2]})})

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp_path / "web"), **k)

        def guess_type(self, path):
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = resolve([f"http://127.0.0.1:{srv.server_port}/data"], tmp_path / "src")
    finally:
        srv.shutdown()
    assert [p.name for p in r.tables] == ["data.csv"]


def test_bad_workbooks_and_xls_are_readable_errors(tmp_path):
    bad = _write(tmp_path / "bad.xlsx", "not a workbook")
    with pytest.raises(SourceError, match="bad.xlsx.*password-protected"):
        resolve([str(bad)], tmp_path / "a")
    broken = tmp_path / "broken.xlsx"
    with zipfile.ZipFile(broken, "w") as z:
        z.writestr("junk.txt", "x")
    with pytest.raises(SourceError, match="could not read the Excel workbook"):
        resolve([str(broken)], tmp_path / "b")
    old = _write(tmp_path / "old.xls", "x")
    with pytest.raises(SourceError, match="save it as .xlsx"):
        resolve([str(old)], tmp_path / "c")


def test_no_table_error_says_what_was_found(tmp_path):
    d = tmp_path / "d"
    _write(d / "model.pkl", "x"); _write(d / "readme.txt", "x"); _write(d / "old.xls", "x")
    with pytest.raises(SourceError) as e:
        resolve([str(d)], tmp_path / "src")
    msg = str(e.value)
    assert "model.pkl: unsupported file type" in msg and "old.xls" in msg and "1 text document" in msg
    assert "xlsx" in msg
