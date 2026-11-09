"""Tests for sim/metric_recorder.py against a real roboeval_metrics.MetricEvaluator.
Needs the simulator venv:  .venv-sim/bin/python -m pytest tests/test_metric_recorder.py
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ev_mod = pytest.importorskip("roboeval_metrics")
from roboeval_metrics import ArmStepData, CollisionContext, ContactPair, MetricConfig, MetricEvaluator, StepData  # noqa: E402

from roboeval_solution.data import score_reward as sr  # noqa: E402
from roboeval_solution.sim.metric_recorder import MetricRecorder, m_run_row, stage_row  # noqa: E402

ROBOT, SCENE, IGNORE = set(range(10)), set(range(10, 30)), set(range(25, 30))
OBJS = ["pot", "lid"]


def make_eval():
    cfg = MetricConfig(track_vel_sync=True, track_vertical_sync=True, track_slippage=True, slip_object_ids=OBJS,
                       slip_sample_window=20, track_collisions=True, is_bimanual=True, arm_ids=["left", "right"])
    ev = MetricEvaluator(cfg)
    init_slip = {(o, a): (bool(np.random.default_rng(len(o)).random() < 0.5), 0.3) for o in OBJS for a in ("left", "right")}
    ev.init(collision_context=CollisionContext(ROBOT, SCENE | IGNORE, IGNORE), initial_slip_state=init_slip)
    return ev, init_slip


def random_sequence(rng, T):
    """Random walks for the arms, contact intervals and hold/release events."""
    seq = []
    pos = {a: rng.normal(size=3) for a in ("left", "right")}
    jp = {a: rng.normal(size=7) for a in ("left", "right")}
    quat = {a: np.r_[1.0, 0, 0, 0] for a in ("left", "right")}
    hold = {(o, a): False for o in OBJS for a in ("left", "right")}
    grip = {a: 0.5 for a in ("left", "right")}
    active = set()
    for t in range(T):
        arms = {}
        for a in ("left", "right"):
            pos[a] = pos[a] + rng.normal(0, 0.01, 3)
            jp[a] = jp[a] + rng.normal(0, 0.02, 7)
            q = quat[a] + rng.normal(0, 0.02, 4)
            quat[a] = q / np.linalg.norm(q)
            grip[a] = float(np.clip(grip[a] + rng.normal(0, 0.05), 0, 1))
            for o in OBJS:
                if rng.random() < 0.08:
                    hold[(o, a)] = not hold[(o, a)]
            arms[a] = ArmStepData(a, jp[a].copy(), rng.normal(0, 0.1, 7), pos[a].copy(), quat[a].copy(), grip[a],
                                  {o: hold[(o, a)] for o in OBJS})
        # contact intervals: pairs appear / disappear
        for _ in range(2):
            if rng.random() < 0.3:
                g1 = int(rng.integers(0, 30))
                g2 = int(rng.integers(0, 30))
                active.add((g1, g2))
            if active and rng.random() < 0.3:
                active.discard(sorted(active)[int(rng.integers(0, len(active)))])
        contacts = [ContactPair(g1, g2, float(rng.choice([-1e-3, 0.0, 1e-9, 5e-3]))) for g1, g2 in sorted(active)]
        seq.append(StepData(arms=arms, contacts=contacts, timestep=0.05))
    return seq


def drive(seq, ev, init_slip, rng=None):
    env = SimpleNamespace(_evaluator=ev)
    rec = MetricRecorder()
    rec.attach(env)
    m_run, flags = [], []
    for t, d in enumerate(seq):
        ev.step(d)
        if rng is not None and rng.random() < 0.05:
            ev.mark_stage(int(rng.integers(1, 5)))
        info = ev.finalize(success_flag=False)
        m_run.append(m_run_row(info, t))
        flags.append(stage_row(info))
    rb, acc = rec.take()
    return {"m_run": np.stack(m_run), "ev_acc": acc, "stage_flags": np.stack(flags), "rb_counts": rb}, info


@pytest.mark.parametrize("seed", range(5))
def test_increments_exact(seed):
    """Column sums of the per-step increments (without RB counts or fail fine) == the evaluator's final metrics."""
    rng = np.random.default_rng(seed)
    T = int(rng.integers(30, 260))
    ev, init_slip = make_eval()
    for j in range(1, 5):
        ev.mark_stage(j, False)
    arrays, final = drive(random_sequence(rng, T), ev, init_slip, rng)
    ep = {"family": "LiftPot", "termination": "budget", "stage_count": 4, "budget_steps": 10 ** 6}
    tot = sr.step_increments({k: v for k, v in arrays.items() if k != "rb_counts"}, ep).sum(0)
    for k in ("cpl", "jpl", "opl", "mjj", "mcj", "bavd", "bgvd"):
        v = final.get(sr.RAW_KEYS[k], 0.0)
        assert abs(tot[sr.MI[k]] - v) < 1e-9 * max(1.0, abs(v)), (k, tot[sr.MI[k]], v)
    for k in ("scc", "ecc", "sc"):
        assert tot[sr.MI[k]] == final[sr.RAW_KEYS[k]], k
    assert abs(tot[sr.MI["tp"]] - final["subtask_progress"]) < 1e-12
    assert abs(tot[sr.MI["ct"]] - T / 20) < 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_rb_counts(seed):
    """Mean over the 20 sampling phases of the official counters == Σ of the RB increments (exactly)."""
    rng = np.random.default_rng(100 + seed)
    T = int(rng.integers(40, 300))
    seq = random_sequence(rng, T)
    ev, init_slip = make_eval()
    arrays, _ = drive(seq, ev, init_slip)
    rb = arrays["rb_counts"].astype(np.float64).sum(0)
    counts = []
    for phase in range(20):
        ev_p, _ = make_eval()
        for d in seq[phase:]:  # phase φ: samples at t = φ, φ+20, …; the first one vs ∅ / the init state
            ev_p.step(d)
        f = ev_p.finalize(success_flag=False)
        counts.append([f["self_collision_count"], f["env_collision_count"], f["slip_count"]])
    mean = np.mean(counts, 0)
    np.testing.assert_allclose(rb, mean, rtol=0, atol=1e-5)
