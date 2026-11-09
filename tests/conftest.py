"""Tests import this checkout's sources (src/ first on sys.path), whichever venv runs them and wherever an editable
install of roboeval_solution in that venv points. The checks against the official scorer, metric evaluator and task
spec need the simulator venv (.venv-sim) and are skipped in the training venv (.venv); README.md has both commands."""
import os
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
if sys.path[:1] != [SRC]:
    sys.path.insert(0, SRC)
# bucket calls are faked in the tests (monkeypatched hf CLI); the name only has to be set
os.environ.setdefault("RBE_BUCKET", "test-user/test-bucket")
