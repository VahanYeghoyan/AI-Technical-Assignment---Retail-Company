"""Pytest configuration.

Puts the project root on sys.path so `import retail_agent` and
`from tests.test_resilience import ...` resolve no matter how pytest is invoked.
Without this, the suite passes under `python -m pytest` (which adds the current
directory) but fails under a bare `pytest` — which is exactly what someone
cloning the repo on another machine will type.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_local_state(tmp_path, monkeypatch):
    """Keep every test's traces and reports out of the developer's var/.

    A Tracer built without an explicit trace_dir falls back to var/traces, so
    each run of the suite used to append a few dozen fake turns to the same
    trace files `/trace` reads.
    """
    monkeypatch.setenv("TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("REPORTS_DB_PATH", str(tmp_path / "reports.db"))
