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

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
