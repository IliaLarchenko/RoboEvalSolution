"""Batched policy for PiModified models (scripts/serve_policy.py).

Per request: transform_one() (input transforms, inpainting anchors -> mask format, per-request inference
params); per batch: launch() (one stacked, asynchronous model call) and finish_batched() (split, output
transforms). infer_batched() runs the three synchronously.
"""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies.policy import Policy
from roboeval_solution.models.observation import Observation
from roboeval_solution.constants import ACTION_HORIZON, ACTION_DIM


class PiModifiedPolicy(Policy):
    """Policy for PiModified models: batched sampling with per-request params, rolling inpainting and
    best-of-N candidates (PiModified.sample_actions returns (actions, {served value key: prediction}))."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        # Re-jit ``sample_actions`` with ``num_candidates`` static (a tiling factor: the post-tile shapes must be known
        # at trace time): one trace per (batch size, num_candidates).
        from openpi.shared import nnx_utils
        self._sample_actions = nnx_utils.module_jit(
            model.sample_actions,
            static_argnames=("num_candidates",),
        )

    def infer_batched(
        self,
        obs_list: list[dict],
        initial_actions_list: list[np.ndarray | None],
        inference_overrides_list: list[dict | None],
        num_candidates: int = 1,
    ) -> list[dict]:
        """Synchronous batched inference: B requests in one model call.

        Args:
            obs_list: one observation dict per request.
            initial_actions_list: inpainting anchors per request (or None).
            inference_overrides_list: per-request inference overrides (or None).
            num_candidates: best-of-N. The VLM prefix runs once on the B requests and its KV cache is tiled to
                B * num_candidates for the flow loop.

        Returns:
            B * num_candidates output dicts; rows [i * N, (i + 1) * N) are request i's candidates.
        """
        return self.finish_batched(self.launch(
            [self.transform_one(o, ia, ov) for o, ia, ov in zip(obs_list, initial_actions_list, inference_overrides_list)],
            num_candidates=num_candidates))

    def transform_one(self, obs: dict, ia: np.ndarray | None, overrides: dict | None) -> dict:
        """Per-request CPU part of a batch (input transforms, inpainting anchors, overrides).
        Independent of the rest of the batch, so a server can run it as requests arrive."""
        action_horizon, action_dim = ACTION_HORIZON, ACTION_DIM
        flat_dim = action_horizon * action_dim
        overrides = overrides or {}

        inp = jax.tree.map(lambda x: x, obs)
        inp = self._input_transform(inp)

        mcfg = self._model.config

        def ov(key, default):
            v = overrides.get(key)
            return default if v is None else v

        out = {
            "inp": inp,
            "tti": float(overrides["time_threshold_inpaint"]),
            "noise_temp": float(overrides["noise_temperature"]),
            "cfg_scale": float(overrides["cfg_scale"]),
            "num_steps": int(ov("num_steps", self._sample_kwargs["num_steps"])),
            # inpaint_snap = hard pin of the anchors (alpha 1); otherwise the soft-pin fraction
            "pin_alpha": 1.0 if int(ov("inpaint_snap", 0)) else float(ov("inpaint_pin_alpha", mcfg.inpaint_pin_alpha)),
            "step_cap": float(ov("inpaint_correction_step_cap", mcfg.inpaint_correction_step_cap)),
            "corr_beta": float(ov("correlation_beta", mcfg.correlation_beta)),
            # best-of-k among the server's candidates (serve_policy --num-candidates), selected after the call
            "num_candidates": int(ov("num_candidates", 1)),
            "hide_velocity": bool(int(ov("hide_velocity", 0))),  # velocity tokens hidden (velocity dropout)
            # cameras dropped (camera dropout), by camera name without "_rgb"
            "hide_cameras": tuple(c for c in ("front", "external") if int(ov(f"hide_{c}", 0))),
            "time_warp": float(ov("time_warp", 1.0)),  # flow time grid t_i = u_i^p
        }

        # Inpainting anchors go through the same input transforms as the training actions
        if ia is not None:
            training_obs = {k: v for k, v in obs.items() if k.startswith("observation/") or k == "task_id"}
            initial_batch = {**training_obs, "actions": np.array(ia, copy=True)}
            transformed = self._input_transform(initial_batch)
            norm_ia = np.asarray(transformed["actions"])  # [n, D]
            clip = mcfg.inpaint_anchor_clip
            if clip:
                norm_ia = np.clip(norm_ia, -clip, clip)

            n = norm_ia.shape[0]
            padded = np.zeros((action_horizon, action_dim), dtype=np.float32)
            padded[:n, :action_dim] = norm_ia[:, :action_dim]
            mask = np.zeros(flat_dim, dtype=np.float32)
            for t in range(n):
                mask[t * action_dim : (t + 1) * action_dim] = 1.0
            out.update(inpaint_target=padded, inpaint_mask=mask, inpaint_length=n)
        else:
            out.update(inpaint_target=np.zeros((action_horizon, action_dim), dtype=np.float32),
                       inpaint_mask=np.zeros(flat_dim, dtype=np.float32), inpaint_length=0)
        return out

    def launch(self, prepped: list[dict], num_candidates: int = 1) -> dict:
        """Stack transform_one() outputs and dispatch the model call asynchronously."""
        B = len(prepped)
        N = max(1, int(num_candidates))
        all_inpaint_lengths = [e["inpaint_length"] for e in prepped]

        # Stack on the host, then one host->device copy per leaf for the whole batch (stacking per-element device
        # arrays costs one small copy per leaf per element). Images stay uint8; Observation.from_dict converts them.
        batched_inputs = jax.device_put(jax.tree.map(
            lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
            *[e["inp"] for e in prepped],
        ))
        inpaint_targets = np.stack([e["inpaint_target"] for e in prepped])  # [B, H, D]
        inpaint_mask = np.stack([e["inpaint_mask"] for e in prepped])       # [B, H * D]
        inpaint_lengths = np.asarray(all_inpaint_lengths, dtype=np.int32)  # [B]
        time_threshold = np.asarray([e["tti"] for e in prepped], dtype=np.float32)  # [B]
        noise_temp = np.asarray([e["noise_temp"] for e in prepped], dtype=np.float32)  # [B]
        cfg_scale_arr = np.asarray([e["cfg_scale"] for e in prepped], dtype=np.float32)  # [B]
        num_steps_arr = np.asarray([e["num_steps"] for e in prepped], dtype=np.int32)  # [B]
        pin_alpha_arr = np.asarray([e["pin_alpha"] for e in prepped], dtype=np.float32)  # [B]
        step_cap_arr = np.asarray([e["step_cap"] for e in prepped], dtype=np.float32)  # [B]
        corr_beta_arr = np.asarray([e["corr_beta"] for e in prepped], dtype=np.float32)  # [B]
        hide_vel_arr = np.asarray([e["hide_velocity"] for e in prepped], dtype=bool)  # [B]
        cams = [k.removesuffix("_rgb") for k in self._model.config.image_keys]
        hide_cam_arr = np.asarray([[c in e["hide_cameras"] for c in cams] for e in prepped], dtype=bool)
        time_warp_arr = np.asarray([e["time_warp"] for e in prepped], dtype=np.float32)  # [B]

        self._rng, sample_rng = jax.random.split(self._rng)

        # The per-request values below replace the server-wide defaults
        sample_kwargs = dict(self._sample_kwargs)

        # Always passed (one JIT trace; all-zero masks are a no-op).
        sample_kwargs["inpaint_targets"] = inpaint_targets
        sample_kwargs["inpaint_mask"] = inpaint_mask
        sample_kwargs["inpaint_lengths"] = inpaint_lengths

        sample_kwargs["time_threshold_inpaint"] = time_threshold
        sample_kwargs["noise_temperature"] = noise_temp
        sample_kwargs["cfg_scale"] = cfg_scale_arr
        sample_kwargs["num_steps"] = num_steps_arr  # per element: the batch runs max(num_steps) iterations
        sample_kwargs["inpaint_pin_alpha"] = pin_alpha_arr
        sample_kwargs["inpaint_correction_step_cap"] = step_cap_arr
        sample_kwargs["hide_velocity"] = hide_vel_arr
        sample_kwargs["hide_cameras"] = hide_cam_arr
        sample_kwargs["time_warp"] = time_warp_arr
        # per-request noise shrinkage (precompute_beta_grid, set by scripts/serve_policy.py)
        sample_kwargs["correlation_beta"] = corr_beta_arr
        sample_kwargs["beta_tables"] = self.beta_tables
        sample_kwargs["num_candidates"] = N

        observation = Observation.from_dict(batched_inputs)
        # (actions [B*N, H, D], {served value key: [B*N, ...]}); row k = request k // N, candidate k % N.
        outs = self._sample_actions(sample_rng, observation, **sample_kwargs)
        return {
            "B": B, "N": N, "batched_inputs": batched_inputs,
            "outs": outs,
        }

    def finish_batched(self, handle: dict) -> list[dict]:
        """Block on the dispatched model call and post-process per request."""
        B, N, batched_inputs = handle["B"], handle["N"], handle["batched_inputs"]
        actions, preds = handle["outs"]
        # One device->host transfer per output instead of one per (request, field).
        actions = np.asarray(actions)
        preds = {k: np.asarray(v) for k, v in preds.items()}

        # Split and output-transform per (request, candidate). State has leading dim B: tile it so each candidate
        # row carries its request's state.
        BN = B * N
        state_tiled = np.repeat(np.asarray(batched_inputs["state"]), N, axis=0)
        results = []
        for i in range(BN):
            outputs_i = {
                "state": state_tiled[i:i+1],
                "actions": actions[i:i+1],
            }
            # The batch-dim strip below applies to every ndim > 0 array: vector predictions (stage / togo /
            # wm_flow_resid) keep a leading batch dim here, scalar predictions are taken 0-d.
            for k, v in preds.items():
                outputs_i[k] = v[i:i+1] if v.ndim >= 2 else v[i]

            outputs_i = {
                k: np.asarray(v[0, ...]) if isinstance(v, (jnp.ndarray, np.ndarray)) and hasattr(v, 'shape') and v.ndim > 0 else v
                for k, v in outputs_i.items()
            }
            outputs_i = self._output_transform(outputs_i)
            results.append(outputs_i)

        return results
