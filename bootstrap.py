"""Resolve local dependencies without changing the existing conda environment."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
HABITAT_LAB = Path(os.environ.get("HABITAT_LAB_PATH", WORKSPACE / "habitat-lab" / "habitat-lab")).expanduser()
DATA = Path(os.environ.get("HABITAT_DATA_PATH", WORKSPACE / "data")).expanduser()
WEIGHTS = Path(os.environ.get("NAV_WEIGHTS_PATH", WORKSPACE / "weights")).expanduser()
HUMANOID_ASSETS = Path(os.environ.get("NAV_HUMANOID_ASSETS", ROOT / "assets" / "female_0")).expanduser()
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "graph_vendor"))
if HABITAT_LAB.exists():
    sys.path.insert(0, str(HABITAT_LAB))
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
