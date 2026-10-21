"""Resolve local dependencies without changing the existing conda environment."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "graph_vendor"))
sys.path.insert(0, str(WORKSPACE / "habitat-lab" / "habitat-lab"))
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
