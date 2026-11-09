"""RoboEval policy I/O transforms: dataset / server sample -> model inputs, model outputs -> server response."""

import dataclasses

import numpy as np

from openpi import transforms

from roboeval_solution.data.value_targets import VALUE_TARGET_KEYS

# Per-sample fields passed through to the model besides state / images / actions: the action pad mask, the
# advantage token input, the task id, the episode step (aux heads' time input; also sent at inference) and the
# value targets (training only).
PASSTHROUGH_KEYS = ("action_is_pad", "advantage", "task_id", "is_demo", "is_scripted", "episode_step",
                    *VALUE_TARGET_KEYS)
# Model outputs returned by the policy server besides the actions (value heads).
OUTPUT_KEYS = ("success_pred", "completion_pred", "stage_pred", "togo_pred", "wm_flow_resid")


@dataclasses.dataclass(frozen=True)
class RoboEvalInputs(transforms.DataTransformFn):
    """`observation/state`, `observation/<camera key>` (uint8 [H, W, 3]; + actions and PASSTHROUGH_KEYS) -> model
    inputs. Allowlist: any other key is dropped."""

    image_keys: tuple[str, ...] = ()

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {k: np.asarray(data[f"observation/{k}"]) for k in self.image_keys},
            "image_mask": {k: np.True_ for k in self.image_keys},
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        for key in PASSTHROUGH_KEYS:
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboEvalOutputs(transforms.DataTransformFn):
    """Model outputs -> actions + the OUTPUT_KEYS value predictions (allowlist)."""

    def __call__(self, data: dict) -> dict:
        result = {"actions": np.asarray(data["actions"])}
        for key in OUTPUT_KEYS:
            if key in data:
                result[key] = data[key]
        return result
