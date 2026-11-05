"""Decisions D4 (banner wording) and D5 (relative paths)."""

from pathlib import Path

from dzmi_app import paths
from dzmi_app.constants import BANNER_TEXT, DATASET_NAME


def test_banner_text_is_the_original_power_bi_wording():
    assert BANNER_TEXT == (
        "SYNTHETIC DATA - demonstration only. "
        "Nothing here describes real Algerian retail demand."
    )
    assert "-" in BANNER_TEXT and "\u2014" not in BANNER_TEXT  # ASCII hyphen, no em dash


def test_project_layout_is_derived_from_the_package_location():
    assert paths.PACKAGE_DIR.name == "dzmi_app"
    assert paths.APP_DIR.name == "streamlit_app"
    assert (paths.PROJECT_ROOT / "streamlit_app").is_dir()
    assert paths.DATA_DIR == paths.PROJECT_ROOT / "data" / "bi_export" / DATASET_NAME


def test_data_folder_exists_and_sits_inside_the_project():
    assert paths.DATA_DIR.is_dir()
    assert paths.PROJECT_ROOT in paths.DATA_DIR.parents


def test_display_path_is_relative_with_forward_slashes():
    shown = paths.display_path(paths.DATA_DIR / "dim_date.csv")
    assert shown == f"data/bi_export/{DATASET_NAME}/dim_date.csv"
    assert not Path(shown).is_absolute()
    assert "\\" not in shown


def test_display_path_never_leaks_paths_outside_the_project(tmp_path):
    outside = tmp_path / "somewhere" / "file.csv"
    assert paths.display_path(outside) == "file.csv"
