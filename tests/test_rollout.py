"""rollout.py coordinator logic without a simulator: the recovery cascade finishes when a worker is lost, error episodes
count as failures, the official seeds are refused. Needs roboeval's challenge package (simulator venv):
    .venv-sim/bin/python -m pytest tests/test_rollout.py
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("roboeval.challenge.scoring")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rollout  # noqa: E402

LOSE = "RBE_TEST_LOSE"  # how the fake worker 0 is lost on its first job: kill (OOM-like) / error (worker_error)


def fake_worker(wid, args, family_queues, results, home_families, param_queues=None, no_more_jobs=None):
    """Stands in for rollout.worker_main: every job fails (a budget episode with zero metrics)."""
    from roboeval.challenge import spec

    if wid != 0 and os.environ.get(LOSE):
        time.sleep(2.0)  # worker 0 takes the first job
    while True:
        job = None
        for q in family_queues.values():
            try:
                job = q.get_nowait()
                break
            except Exception:  # noqa: BLE001 - queue.Empty
                continue
        if job is None:
            if no_more_jobs.is_set() and all(q.qsize() == 0 for q in family_queues.values()):
                return
            time.sleep(0.05)
            continue
        if wid == 0 and os.environ.get(LOSE) == "kill":
            os.kill(os.getpid(), signal.SIGKILL)
        if wid == 0 and os.environ.get(LOSE) == "error":
            results.put({"worker_error": "boom", "worker": wid})
            return
        time.sleep(0.05)
        results.put({**{k: v for k, v in job.items() if k != "prefix"}, "termination": "budget", "success": False,
                     "steps": 10, "metrics": {k: 0.0 for k in spec.METRIC_KEYS}, "score_ep": {"score": 0.0}})


def _collect(root: Path, n: int) -> Path:
    from roboeval_solution.data.store import StoreWriter

    w = StoreWriter(root / "stores" / "w00", cameras=[], info={"source": "rollout"})
    for i in range(n):  # failures with a P̂(success) drop at frame 70 (no stall event), 3 per seed
        T = 120
        ps = np.r_[np.full(70, 0.8), np.full(T - 70, 0.2)].astype(np.float32)
        actions = np.zeros((T, 14), np.float32)
        actions[:, 0] = 0.001  # the left hand moves, grippers open: no stall event
        w.add_episode({"actions": actions, "pred_success": ps,
                       "pred_stage": np.full((T, 5), 0.1, np.float32), "stage_flags": np.zeros((T, 5), np.uint8),
                       "sc": np.zeros(T, np.int32), "state_gripper": np.ones((T, 2), np.float32),
                       "state_ee": np.zeros((T, 12), np.float32)}, {},
                      {"family": "LiftPot", "variation": "PositionAndOrientation", "seed": 20_000_000 + i // 3,
                       "success": False, "termination": "fail", "tp": 0.0, "metrics": {"tp": 0.0},
                       "score_ep": {"score": 0.0},
                       "recording_ok": True, "budget_steps": 200, "stage_count": 4})
    w.close()
    return root


@pytest.mark.parametrize("lose", ["", "kill", "error"])
def test_cascade_finishes_when_a_worker_is_lost(tmp_path, monkeypatch, lose):
    col = _collect(tmp_path / "c0_0001_step_000500", n=6)
    out = tmp_path / "rec"
    monkeypatch.setenv(LOSE, lose)
    monkeypatch.setattr(rollout, "worker_main", fake_worker)
    monkeypatch.setattr(sys, "argv", ["rollout.py", "--recover", str(col), "--out", str(out), "--workers", "2"])
    t0 = time.time()
    if lose:
        with pytest.raises(SystemExit) as e:
            rollout.main()
        assert e.value.code == 2  # cut short: summary written, marked partial
    else:
        rollout.main()
    assert time.time() - t0 < 120
    summary = json.loads((out / "summary.json").read_text())
    eps = [json.loads(line) for line in open(out / "episodes.jsonl")]
    if lose:
        assert summary["partial"]["missing"] >= 1 and summary["partial"]["episodes"] == len(eps)
    else:
        # every stage of both chains (drop,perturb,reverse,peak,back2s | retry) applies to each of the 6 targets
        assert "partial" not in summary and len(eps) == 36 and summary["recovery"]["exhausted"] == 6
    # every retry of a target has its own parameter seed (several targets share an env seed in a collect)
    seeds = [e["script_seed"] for e in eps if e["scenario"] == "retry"]
    assert len(seeds) == len(set(seeds))


def test_error_episodes_count_as_failures():
    from roboeval.challenge import spec

    slot = object.__new__(rollout.Slot)
    slot.job = {"family": "LiftPot", "variation": "PositionAndOrientation", "seed": 20_000_001, "prefix": np.zeros(3)}
    slot.t, slot.info = 25, {"success": 1.0, "subtask_progress": 0.5, "total_cartesian_path_length": 1.5}
    err = slot.error_result("Traceback ... boom")
    assert err["termination"] == "error" and not err["success"] and "prefix" not in err
    assert err["metrics"]["success"] == 0.0 and err["metrics"]["ct"] == 25 / 20 and err["metrics"]["tp"] == 0.5
    ok = {"family": "LiftPot", "termination": "success", "success": True,
          "metrics": {**{k: 0.0 for k in spec.METRIC_KEYS}, "success": 1.0, "tp": 1.0}}
    bare = {"family": "LiftPot", "termination": "error", "success": False}  # no metrics: zeros
    s = rollout.summarize([ok, err, bare])
    assert s["counts"]["LiftPot"] == {"episodes": 3, "errors": 2}
    assert np.isclose(s["per_family"]["LiftPot"]["success"], 1 / 3)


def test_refuses_official_seeds_and_param_search_with_recover(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rollout.py", "--out", str(tmp_path / "x"), "--seed-start", "1000"])
    with pytest.raises(SystemExit, match="official evaluation seeds"):
        rollout.main()
    monkeypatch.setattr(sys, "argv", ["rollout.py", "--out", str(tmp_path / "y"), "--recover", str(tmp_path),
                                      "--param-search", str(tmp_path / "ps.json")])
    with pytest.raises(SystemExit) as e:
        rollout.main()
    assert e.value.code == 2  # argparse error


def test_stall_boost():
    """stall_boost (tuning/stall.StallDetector "old" events in rollout.Slot): a success-prediction drop of 0.2 below
    its high, or 6 flat calls below 0.5, boost the next 3 policy calls (1.5x noise temperature, every server
    candidate); the episode's own overrides otherwise; setting 0 never boosts."""
    from types import SimpleNamespace

    import numpy as np
    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import rollout
    from roboeval_solution.tuning.stall import StallDetector

    r = SimpleNamespace(job={"overrides": {"noise_temperature": 0.6}}, det=StallDetector("LiftPot"), stall_mode=1,
                        boost_left=0, n_boost=0, t=0)

    def check(v):
        rollout.Slot._stall_events(r, r.det.on_call({"success_pred": np.float32([v])}, r.t, r.boost_left > 0))

    def overrides():
        return rollout.Slot._overrides(r)

    for v in (0.3, 0.6, 0.7):
        check(v)
    assert overrides() == {"noise_temperature": 0.6}
    check(0.45)  # 0.25 below the high
    for _ in range(3):
        ov = overrides()
        assert ov["noise_temperature"] == pytest.approx(0.9) and ov["num_candidates"] == 64
    assert overrides() == {"noise_temperature": 0.6} and r.n_boost == 3
    for _ in range(5):
        check(0.3)
    assert r.boost_left == 0
    check(0.3)  # 6 flat calls below 0.5 since the last boost
    assert r.boost_left == 3
    r0 = SimpleNamespace(job={"overrides": {}}, det=StallDetector("LiftPot"), stall_mode=0, boost_left=0, n_boost=0)
    for v in (0.9, 0.2):
        rollout.Slot._stall_events(r0, r0.det.on_call({"success_pred": np.float32([v])}, 0, False))
    assert r0.boost_left == 0 and r0.det.first  # detected, not boosted
