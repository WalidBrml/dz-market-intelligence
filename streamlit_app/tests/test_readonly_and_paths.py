"""The app is read-only (requirement) and uses only relative paths (decision D5)."""

import ast
import hashlib
import re
from pathlib import Path

import pytest

from dzmi_app.data_loader import clear_cache, load_dataset
from dzmi_app import paths
from oracle_values import MANIFEST_ROWS

WRITE_METHODS = {
    "to_csv", "to_parquet", "to_excel", "to_sql", "to_json", "to_pickle", "to_feather",
    "write_text", "write_bytes", "unlink", "rmdir", "rmtree", "mkdir", "touch", "file_uploader",
}
FORBIDDEN_IMPORTS = {"shutil", "sqlite3", "tempfile"}
WRITE_MODE_CHARS = set("wax+")

# Built by concatenation so that this file does not match its own patterns.
DRIVE_LETTER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]" + r":[\\/]")
USER_DIRS = re.compile(r"[\\/](?:Users|home)[\\/]|" + "DE" + "LL")


def find_write_calls(source: str) -> list[str]:
    """Names of calls that could write to disk, plus forbidden imports."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
                found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in WRITE_METHODS:
                found.append(f.attr)
            mode = None
            if isinstance(f, ast.Name) and f.id == "open":
                mode = node.args[1] if len(node.args) > 1 else None
            elif isinstance(f, ast.Attribute) and f.attr == "open":
                mode = node.args[0] if node.args else None
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode = kw.value
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str) \
                    and WRITE_MODE_CHARS & set(mode.value):
                found.append(f"open(mode={mode.value!r})")
    return found


def app_sources(app_dir: Path) -> list[Path]:
    """All application sources (the tests folder is excluded)."""
    return [p for p in sorted(app_dir.rglob("*.py")) if "tests" not in p.relative_to(app_dir).parts]


# ------------------------------------------------------------------ the detector itself


@pytest.mark.parametrize("snippet", [
    "open('x.csv', 'w')", "open('x.csv', mode='a')", "open('x', 'r+')",
    "import shutil", "from sqlite3 import connect", "df.to_csv('x')", "df.to_parquet('x')",
    "p.write_text('x')", "p.open('wb')", "st.file_uploader('x')", "os.unlink('x')",
])
def test_detector_flags_write_operations(snippet):
    assert find_write_calls(snippet)


@pytest.mark.parametrize("snippet", [
    "open('x.csv')", "open('x.csv', 'r')", "open('x', mode='rb')", "pd.read_csv('x')",
    "p.read_text()", "text.replace('a', 'b')", "import pandas as pd",
])
def test_detector_ignores_read_operations(snippet):
    assert find_write_calls(snippet) == []


# ------------------------------------------------------------------ the application


def test_application_sources_exist(app_dir):
    names = {p.name for p in app_sources(app_dir)}
    assert {"app.py", "paths.py", "data_loader.py", "dims_forecast.py", "integrity.py"} <= names


def test_application_contains_no_write_operations(app_dir):
    offenders = {}
    for path in app_sources(app_dir):
        calls = find_write_calls(path.read_text(encoding="utf-8"))
        if calls:
            offenders[path.relative_to(app_dir).as_posix()] = calls
    assert offenders == {}


def test_application_contains_no_absolute_or_user_specific_paths(app_dir):
    offenders = []
    for path in app_sources(app_dir) + sorted(app_dir.glob("*.md")) + sorted(app_dir.glob("*.txt")) \
            + sorted(app_dir.glob("*.ini")) + sorted((app_dir / ".streamlit").glob("*.toml")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if DRIVE_LETTER.search(line) or USER_DIRS.search(line):
                offenders.append(f"{path.relative_to(app_dir).as_posix()}:{n}")
    assert offenders == []


def test_path_detector_catches_local_paths():
    assert DRIVE_LETTER.search("C" + ":" + chr(92) + "data")
    assert DRIVE_LETTER.search("d:/data")
    assert USER_DIRS.search("/home/someone/x")
    assert not DRIVE_LETTER.search("data/bi_export/smoke_m3_20261025")
    assert not DRIVE_LETTER.search("model:float?")      # schema mini-language must not match


def test_no_path_constant_is_absolute():
    # Resolved at runtime from __file__, but nothing hard-coded: the dataset name is a bare name.
    from dzmi_app.constants import DATASET_NAME, MANIFEST_FILE
    for value in (DATASET_NAME, MANIFEST_FILE):
        assert "/" not in value and chr(92) not in value and ":" not in value


def _digests(folder: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())
            if p.is_file()}


def test_loading_and_checking_never_modifies_the_data_folder():
    from dzmi_app.integrity import run_integrity_checks
    before = _digests(paths.DATA_DIR)
    expected_files = {f"{t}.csv" for t in MANIFEST_ROWS} | {"bi_export_manifest.json"}
    assert expected_files <= set(before)         # 22 CSV exports + the manifest
    names_before = set(before)
    clear_cache()
    ds = load_dataset()
    run_integrity_checks(ds)
    after = _digests(paths.DATA_DIR)
    assert set(after) == names_before
    assert after == before
