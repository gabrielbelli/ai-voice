"""Make the sample service importable the way a consumer's own package is."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
