"""serve_policy's pipelined batching with a stub policy (no model): every request is answered, best-of-k picks the
candidate with the highest predicted Δsuccess among the request's first k, buckets must include 1."""
import asyncio
import sys
import threading
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
serve_policy = pytest.importorskip("serve_policy")

H, N = 30, 3


class StubPolicy:
    """Candidate c of every request: actions filled with c, predicted Δsuccess (wm_flow_resid[0]) = SCORES[c]."""
    SCORES = (0.1, 0.5, 0.3)
    STAGE = (0.0, 0.0, 1.0)  # predicted Δ of the first stage: only candidate 2 advances it

    def transform_one(self, obs, initial, overrides):
        return {"num_candidates": int(overrides.get("num_candidates", 1)), "state": obs["observation/state"]}

    def launch(self, prepped, num_candidates):
        return {"outs": np.zeros(1), "B": len(prepped), "N": num_candidates}

    def finish_batched(self, handle):
        return [{"actions": np.full((H, 14), c, np.float32), "success_pred": np.float32(0.7),
                 "wm_flow_resid": np.r_[self.SCORES[c], self.STAGE[c], np.zeros(15)].astype(np.float32)}
                for _ in range(handle["B"]) for c in range(handle["N"])]


def make_server():
    bp = object.__new__(serve_policy.BatchedPolicy)
    bp.data = SimpleNamespace(camera_map=())
    bp.policy, bp.num_candidates, bp.H, bp.max_batch, bp.buckets = StubPolicy(), N, H, 8, [1, 4, 8]
    bp.defaults, bp.cam_to_key, bp.clients = {}, {}, 0
    bp.rank_coef = {0: np.r_[1.0, 0.5, np.zeros(15)]}  # task 0: Δsuccess + 0.5 x Δstage_1
    bp.q, bp._inflight, bp._gpu_free = Queue(), Queue(), threading.Event()
    bp.stats = {"requests": 0, "batches": 0, "infer_s": 0.0, "gpu_busy_s": 0.0}
    bp._gpu_free.set()
    threading.Thread(target=bp._dispatch_loop, daemon=True).start()
    threading.Thread(target=bp._finish_loop, daemon=True).start()
    return bp


def test_pipelined_batches_and_best_of_k():
    bp = make_server()
    req = {"proprio": np.zeros(36, np.float32), "grippers": np.zeros(2, np.float32), "task_id": 0, "images": {},
           "episode_step": 5, "inpaint": None}

    async def run():
        return await asyncio.gather(*[bp.infer({**req, "overrides": {"num_candidates": k}}) for k in (1, 2, 3, 3, 1)])

    resps = asyncio.run(asyncio.wait_for(run(), 30))
    chosen = [int(r["actions_model"][0, 0]) for r in resps]
    assert chosen == [0, 1, 1, 1, 0]  # k = 1: the plain sample; k >= 2: the best Δsuccess of the first k
    assert all(r["values"]["success_pred"] == np.float32(0.7) for r in resps)
    assert bp.stats["requests"] == 5


def test_rank_full_uses_the_score_weighted_residual():
    bp = make_server()
    req = {"proprio": np.zeros(36, np.float32), "grippers": np.zeros(2, np.float32), "images": {},
           "episode_step": 5, "inpaint": None}

    async def run():
        return await asyncio.gather(*[bp.infer({**req, "task_id": t, "overrides": {"num_candidates": 3, "rank_full": r}})
                                      for t, r in ((0, 0), (0, 1), (5, 1))])

    resps = asyncio.run(asyncio.wait_for(run(), 30))
    # Δsuccess alone: candidate 1 (0.5); score-weighted: candidate 2 (0.3 + 0.5 x 1.0); no weights for task 5: Δsuccess
    assert [int(r["actions_model"][0, 0]) for r in resps] == [1, 2, 1]


def test_rank_coefficients(tmp_path):
    from roboeval_solution.data import score_reward as sr
    from roboeval_solution.data.tasks import TASK_TO_ID

    g = np.zeros(13); g[sr.MI["success"]], g[sr.MI["tp"]], g[sr.MI["ct"]] = 0.6, 0.3, -0.1
    (tmp_path / "score_weights.json").write_text(__import__("json").dumps({"round": 1, "g": {"StackTwoBlocks": g.tolist()}}))
    c = serve_policy.rank_coefficients(tmp_path / "score_weights.json")[TASK_TO_ID["StackTwoBlocks"]]
    assert c[0] == pytest.approx(0.9) and np.allclose(c[1:4], 0.1) and np.all(c[4:6] == 0)  # 3 stages
    assert c[6] == pytest.approx(-0.1 * sr.togo_scale("StackTwoBlocks")[0])  # time-to-go
    assert serve_policy.rank_coefficients(tmp_path / "missing.json") == {}


def test_buckets_must_include_one(monkeypatch):
    monkeypatch.setattr(serve_policy.policy_config, "create_trained_policy", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("bucket check comes first")))
    cfg = SimpleNamespace(data=SimpleNamespace(camera_map=()), model=SimpleNamespace(action_horizon=H))
    with pytest.raises(ValueError, match="include 1"):
        serve_policy.BatchedPolicy(cfg, "/nonexistent", 8, {}, buckets=(4, 8))
