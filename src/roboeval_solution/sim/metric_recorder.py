"""Read-only observer of the env's roboeval_metrics.MetricEvaluator.

Every env step, `MetricRolloutEval._metric_step` feeds the evaluator a StepData (per-arm kinematics, gripper
holding state, all contacts). The recorder wraps `env._evaluator.step` after every reset (each reset builds a
new evaluator), lets the official update run first and then only reads:

* `ev_acc` [8]: evaluator accumulators after the step (jerk sums / sample counts, bimanual sums, frame),
  so jerk and coordination averages decompose exactly into per-step increments (score_reward.step_increments);
* Rao-Blackwellised collision / slip increments [3]: the official counters sample every 20 frames,
  comparing with the previous sample; averaging over a uniformly random sampling phase gives the exact
  per-frame formula  |R_t \\ R_{t−20}| / 20  (R_{t<0} = ∅, slips vs the init state) with the same expectation
  and much lower variance.

Nothing in the simulation changes.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from roboeval_solution.data.score_reward import MAX_STAGES, RAW_KEYS, METRIC13

_MARGIN = 1e-8  # roboeval_metrics.evaluator._DEFAULT_COLLISION_MARGIN


class MetricRecorder:
    def __init__(self):
        self.ev = None

    # ── lifecycle ──
    def attach(self, env) -> None:
        """Call right after every env.reset(seed)."""
        ev = env._evaluator
        self.reset(ev)
        if getattr(ev.step, "_rbe_observer", None) is self:
            return
        orig = ev.step

        def step(data, _orig=orig, _ev=ev):
            _orig(data)  # official update first (semantics unchanged)
            self.observe(_ev, data)

        step._rbe_observer = self
        ev.step = step

    def reset(self, ev) -> None:
        self.ev = ev
        cfg = ev._cfg
        self.window = max(1, int(cfg.slip_sample_window))
        self.track_coll = bool(cfg.track_collisions)
        self.track_slip = bool(cfg.track_slippage)
        self.slip_ids = list(cfg.slip_object_ids)
        if self.track_coll:
            ctx = ev._coll_ctx
            self.robot_ids = set(ctx.robot_geom_ids)
            self.scene_ids = set(ctx.scene_geom_ids)
            self.external = self.scene_ids - set(ctx.ignore_geom_ids)
        self.init_slip = dict(ev._prev_slip_state)  # (obj, arm) -> (holding, gripper opening) at reset
        self.ring: deque = deque(maxlen=self.window)  # past samples (R, E, slip state)
        self.rb: list = []
        self.acc: list = []

    # ── per step ──
    def _sets(self, data):
        R, E = set(), set()
        if self.track_coll:
            for c in data.contacts:
                if c.distance > _MARGIN:
                    continue
                g1, g2 = c.geom1, c.geom2
                o1 = "robot" if g1 in self.robot_ids else ("scene" if g1 in self.scene_ids else "other")
                o2 = "robot" if g2 in self.robot_ids else ("scene" if g2 in self.scene_ids else "other")
                if o1 == o2 == "robot":
                    R.add((g1, g2) if g1 <= g2 else (g2, g1))
                elif {o1, o2} == {"robot", "scene"}:
                    s = g1 if o1 == "scene" else g2
                    if s in self.external:
                        E.add(s)
        H = {}
        if self.track_slip:
            for obj in self.slip_ids:
                for arm_id, arm in data.arms.items():
                    H[(obj, arm_id)] = (bool(arm.holding_object.get(obj, False)), float(arm.gripper_opening))
        return R, E, H

    def observe(self, ev, data) -> None:
        R, E, H = self._sets(data)
        if len(self.ring) == self.window:  # sample 20 frames ago
            R0, E0, H0 = self.ring[0]
        else:
            R0, E0, H0 = set(), set(), self.init_slip
        n = self.window
        scc = len(R - R0) / n
        ecc = len(E - E0) / n
        slips = 0
        for key, (hold, q) in H.items():
            if key in H0:
                h0, q0 = H0[key]
                if h0 and not hold and q <= q0:
                    slips += 1
        self.rb.append((scc, ecc, slips / n))
        self.ring.append((R, E, H))
        a = ev
        self.acc.append((
            sum(a._joint_jerk_sum.values()), sum(a._cartesian_jerk_sum.values()), sum(a._jerk_sample_count.values()),
            a._sum_vel_err, a._n_vel, a._sum_z_err, a._n_z, a._frame,
        ))

    # ── outputs ──
    def take(self) -> tuple[np.ndarray, np.ndarray]:
        """(rb_counts [T,3] f32, ev_acc [T,8] f64) for the steps observed since attach()."""
        rb = np.asarray(self.rb, np.float32).reshape(-1, 3)
        acc = np.asarray(self.acc, np.float64).reshape(-1, 8)
        return rb, acc


def m_run_row(info: dict, t: int) -> np.ndarray:
    """Running official metrics after action t (METRIC13 order) from the env's step info."""
    row = np.zeros(len(METRIC13), np.float64)  # f64: path sums must telescope to 1e-9
    for i, k in enumerate(METRIC13):
        if k == "ct":
            row[i] = (t + 1) / 20.0
            continue
        v = info.get(RAW_KEYS[k])
        row[i] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) and v is not None else 0.0
    return row


def stage_row(info: dict) -> np.ndarray:
    """Stage flags 1..J from info["task_stage_reached"], zero-padded to MAX_STAGES."""
    flags = np.zeros(MAX_STAGES, np.uint8)
    for k, v in (info.get("task_stage_reached") or {}).items():
        j = int(k) - 1
        if 0 <= j < MAX_STAGES:
            flags[j] = 1 if v else 0
    return flags
