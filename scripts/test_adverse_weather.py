"""Adverse-weather test entry point (same as ``python main.py weather``).

    python scripts/test_adverse_weather.py --weather fog
    python scripts/test_adverse_weather.py --weather rain
    python scripts/test_adverse_weather.py --weather all
    python scripts/test_adverse_weather.py --source real

See evaluation/adverse_weather.py and docs/ADVERSE_WEATHER.md.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "evaluation" / "adverse_weather.py"), run_name="__main__")
