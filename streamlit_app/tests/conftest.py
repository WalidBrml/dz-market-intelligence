import sys
from pathlib import Path

import pytest

# Make "dzmi_app" importable even if pytest is launched without pytest.ini's pythonpath.
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dzmi_app.data_loader import load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def overview_context():
    from dzmi_app.overview import load_overview_context
    return load_overview_context()


@pytest.fixture(scope="session")
def app_dir():
    return APP_DIR
