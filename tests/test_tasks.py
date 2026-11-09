"""data/tasks.py (task ids, budgets) against the official spec and the other copies of the same facts."""
import pytest

from roboeval_solution.data import score_reward as sr
from roboeval_solution.data import tasks


def test_copies_agree():
    assert sr.BUDGET_SECONDS == tasks.BUDGET_SECONDS
    assert set(sr.STAGE_COUNT) == set(tasks.TASK_FAMILIES)
    assert tasks.NUM_TASKS == len(tasks.TASK_FAMILIES) == len(tasks.TASK_TO_ID)
    assert tasks.split_variation_name("CubeHandoverPositionAndOrientation") == ("CubeHandover", "PositionAndOrientation")
    assert tasks.split_variation_name("PickSingleBookFromTable") == ("PickSingleBookFromTable", "")


def test_matches_official_spec():
    spec = pytest.importorskip("roboeval.challenge.spec")  # simulator venv only
    # the task id is a model input: the order must be the official one
    assert tasks.TASK_FAMILIES == tuple(f.key for f in spec.TASK_FAMILIES)
    assert tasks.BUDGET_SECONDS == {f.key: f.budget_seconds for f in spec.TASK_FAMILIES}
