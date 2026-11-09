"""Project-wide constants."""

# Number of task families (roboeval_solution.data.tasks.TASK_FAMILIES, fixed order: Observation.task_id indexes it for
# the task input token and the task head).
from roboeval_solution.data.tasks import NUM_TASKS  # noqa: E402,F401

# Action space
ACTION_DIM = 14  # ee-delta: 2 x (3 pos + 3 rotvec) + 2 grippers
ACTION_HORIZON = 30  # 1.5 seconds at 20 FPS

# Togo head width: one normalised behaviour-to-go output per data.score_reward.TOGO_KEYS entry, shared by all tasks.
from roboeval_solution.data.score_reward import MAX_STAGES, TOGO_KEYS, WM_VECTOR_DIM  # noqa: E402
TOGO_HEAD_WIDTH = len(TOGO_KEYS)
assert WM_VECTOR_DIM == 1 + MAX_STAGES + TOGO_HEAD_WIDTH  # WM-flow ψ = [success, stage flags x5, togo]
