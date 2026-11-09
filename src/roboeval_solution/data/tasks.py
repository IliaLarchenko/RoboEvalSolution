"""Task families of the RoboEval challenge and their ids (the conditioning token index)."""

# Canonical order = roboeval.challenge.spec.TASK_FAMILIES. Do not reorder: the id is a model input.
TASK_FAMILIES = (
    "LiftPot",
    "LiftTray",
    "StackTwoBlocks",
    "StackSingleBookShelf",
    "CubeHandover",
    "PackBox",
    "PickSingleBookFromTable",
    "RotateValve",
)
TASK_TO_ID = {name: i for i, name in enumerate(TASK_FAMILIES)}
NUM_TASKS = len(TASK_FAMILIES)

VARIATIONS = ("PositionAndOrientation", "Position", "Orientation", "")

# Evaluation budget (simulated seconds) per family, from roboeval/challenge/spec.py (tests/test_tasks.py checks it,
# and that data/score_reward.py's copy agrees).
BUDGET_SECONDS = {
    "LiftPot": 10.0,
    "LiftTray": 10.0,
    "StackTwoBlocks": 20.0,
    "StackSingleBookShelf": 40.0,
    "CubeHandover": 20.0,
    "PackBox": 20.0,
    "PickSingleBookFromTable": 15.0,
    "RotateValve": 20.0,
}


def split_variation_name(name: str) -> tuple[str, str]:
    """'CubeHandoverPositionAndOrientation' -> ('CubeHandover', 'PositionAndOrientation')."""
    for fam in sorted(TASK_FAMILIES, key=len, reverse=True):
        if name.startswith(fam):
            rest = name[len(fam):]
            if rest in VARIATIONS:
                return fam, rest
    raise ValueError(f"Unknown task variation name: {name}")
