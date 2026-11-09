"""Batched websocket policy server for RoboEval rollouts.

    .venv/bin/python scripts/serve_policy.py --config-name pi05_roboeval --checkpoint-dir <checkpoint> --port 8765

Protocol (openpi_client msgpack_numpy frames, one request -> one response):
  request  {"proprio": f32[36], "grippers": f32[2], "images": {cam: u8[H,W,3]}, "task_id": int,
            "episode_step": int (env steps since the reset; the value / Q heads' time input, 0 if absent),
            "inpaint": None | {"chunk": f32[H,14] (model repr), "executed": int, "keep": int},
            "overrides": {num_steps, noise_temperature, time_threshold_inpaint, cfg_scale, inpaint_snap,
                          inpaint_pin_alpha, inpaint_correction_step_cap, correlation_beta, num_candidates,
                          hide_velocity, hide_front, hide_external, time_warp, rank_full}}
           (all per request: mixed values share a batch; a batch runs max(num_steps) flow steps)
  response {"actions_model": f32[H,14], "actions_env": f32[H,14], "values": {VALUE_KEYS}, "infer_ms": float,
            "batch": int, "best_of_n": {...} (only when k > 1 candidates were ranked)}
  {"type": "ping"} -> {"ok": true, "metadata": {...}}

All representation logic lives here (state construction, model<->env action repr,
inpainting re-basing), so rollout workers only send raw env observations.
Requests from all connected workers are batched: each request is transformed as it arrives, and
whenever the GPU frees up everything prepared is launched as one forward pass (rounded down to a
compiled batch size), while the previous batch is post-processed.
"""
from __future__ import annotations

import argparse
import json
import asyncio
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from queue import Empty, Queue

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import numpy as np
import websockets.asyncio.server as ws_server
from openpi_client import msgpack_numpy

from roboeval_solution.data import action_repr as ar
from roboeval_solution.policies import policy_config
from roboeval_solution.training.config import get_config

logger = logging.getLogger("serve")

# Value-head outputs forwarded to rollout workers (data/value_targets.py):
# success_pred = P(success), completion_pred, stage_pred [5] = P(stage j reached by the end), togo_pred [11] =
# normalised behaviour-to-go, wm_flow_resid [17] = Q − V residual [Δsucc, Δstage x5, Δtogo x11] of the sampled chunk
# (conditional pass).
VALUE_KEYS = ("success_pred", "completion_pred", "stage_pred", "togo_pred", "wm_flow_resid")
PER_REQUEST_OVERRIDES = ("num_steps", "noise_temperature", "time_threshold_inpaint", "cfg_scale", "inpaint_snap",
                         "inpaint_pin_alpha", "inpaint_correction_step_cap", "correlation_beta", "num_candidates",
                         "hide_velocity", "hide_front", "hide_external", "time_warp", "rank_full")
# Correlated-noise shrinkage values a request can choose (correlation_beta snaps to the nearest).
BETA_GRID = tuple(round(0.1 * i, 1) for i in range(10))


def rank_coefficients(path: Path) -> dict[int, np.ndarray]:
    """{task id: [17] weights of the WM-flow Q - V residual [Δsucc, Δstage x5, Δtogo x11]} giving the change of the
    score-weighted value (score_reward.value_blocks, leaderboard-exact weights g of the export's score_weights.json):
    Δsucc x (g_success + 0.3 completion potential), Δstage_j x g_tp / J (j < J), Δtogo_k x g_k x scale_k. {} without
    the file (rank_full then ranks by Δsuccess)."""
    if not path.exists():
        logger.warning("no %s: rank_full ranks candidates by Δsuccess", path)
        return {}
    from roboeval_solution.data import score_reward as sr
    from roboeval_solution.data.tasks import TASK_TO_ID

    out = {}
    for fam, g in json.loads(path.read_text())["g"].items():
        if fam not in TASK_TO_ID:
            continue
        g = np.asarray(g, np.float64)
        J = sr.STAGE_COUNT[fam]
        c = np.zeros(sr.WM_VECTOR_DIM)
        c[0] = g[sr.MI["success"]] + 0.3
        c[1:1 + J] = g[sr.MI["tp"]] / J
        c[1 + sr.MAX_STAGES:] = np.array([g[sr.MI[k]] for k in sr.TOGO_METRIC]) * sr.togo_scale(fam)
        out[TASK_TO_ID[fam]] = c
    logger.info("rank_full: score-weighted candidate ranking for %d tasks", len(out))
    return out


class BatchedPolicy:
    def __init__(self, cfg, ckpt_dir: str, max_batch: int, defaults: dict,
                 buckets: tuple | None = None, num_candidates: int = 1):
        self.cfg = cfg
        self.max_batch = max_batch
        # Compiled batch sizes. Batches are rounded DOWN to a bucket (leftovers ride the next batch), so no
        # padding work is wasted; each bucket costs one warmup compile.
        buckets = buckets or (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
        self.buckets = [b for b in buckets if b < max_batch] + [max_batch]
        if self.buckets[0] != 1:  # the dispatcher rounds a backlog DOWN to a bucket: a lone request needs 1
            raise ValueError(f"batch buckets must include 1: {self.buckets}")
        # best-of-N: every request gets N flow samples off one shared prefix pass; its num_candidates k
        # (<= N) picks the best of its first k by the WM-flow Q head (_select; k = 1: plain sample)
        self.num_candidates = max(1, int(num_candidates))
        self.data = cfg.data
        self.ckpt_dir = str(ckpt_dir)
        # identity of the weights (copies / links of one export share it): the export's source checkpoint
        info = Path(ckpt_dir) / "EXPORT_INFO"
        src = [l.split("=", 1)[1].strip() for l in info.read_text().splitlines() if l.startswith("source=")] \
            if info.exists() else []
        self.policy_id = src[0] if src else str(Path(ckpt_dir).resolve())
        self.policy = policy_config.create_trained_policy(
            cfg, ckpt_dir, sample_kwargs={"num_steps": int(defaults.pop("num_steps", 10))})
        self.num_steps = self.policy._sample_kwargs["num_steps"]
        # per-request correlated-noise shrinkage
        self.policy.beta_tables = self.policy._model.precompute_beta_grid(BETA_GRID)
        self.H = cfg.model.action_horizon
        self.rank_coef = rank_coefficients(Path(ckpt_dir) / "score_weights.json")
        self.clients = 0
        self.defaults = defaults
        self.cam_to_key = {cam: key for key, cam in self.data.camera_map}
        self.q: Queue = Queue()
        self.stats = {"requests": 0, "batches": 0, "infer_s": 0.0, "gpu_busy_s": 0.0}
        self.warmup()
        self._inflight: Queue = Queue()
        self._gpu_free = threading.Event()
        self._gpu_free.set()
        threading.Thread(target=self._dispatch_loop, daemon=True).start()
        threading.Thread(target=self._finish_loop, daemon=True).start()

    def metadata(self) -> dict:
        return {
            "config": self.cfg.name,
            "action_horizon": self.H,
            "cameras": [cam for _, cam in self.data.camera_map],
            "defaults": self.defaults,
            "num_steps": self.num_steps,
            "per_request_overrides": list(PER_REQUEST_OVERRIDES),
            "max_candidates": self.num_candidates,
            "checkpoint": self.ckpt_dir,
            "policy_id": self.policy_id,
        }

    def _prepare(self, req: dict):
        obs = {
            "observation/state": ar.make_state(req["proprio"], req["grippers"]),
            "task_id": np.int32(req["task_id"]),
            "episode_step": np.float32(req.get("episode_step", 0)),
        }
        for cam, img in req["images"].items():
            if cam in self.cam_to_key:
                img = np.asarray(img)
                if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
                    img = np.transpose(img, (1, 2, 0))
                obs[f"observation/{self.cam_to_key[cam]}"] = img
        initial = None
        inp = req.get("inpaint")
        if inp is not None and int(inp.get("keep", 0)) > 0:
            chunk, k, keep = np.asarray(inp["chunk"], np.float32), int(inp["executed"]), int(inp["keep"])
            tail = ar.rebase_cumulative(chunk, k)
            initial = tail[:keep] if len(tail) >= 1 else None
        overrides = dict(self.defaults)
        overrides.update(req.get("overrides") or {})
        return obs, initial, overrides

    @staticmethod
    def _select(cands: list, k: int, coef: np.ndarray | None = None) -> tuple[dict, dict]:
        """Best of the first k candidates by the WM-flow Q head: its Δsuccess (wm_flow_resid[0]), or with `coef` the
        score-weighted Q - V residual coef · wm_flow_resid (rank_coefficients)."""
        k = max(1, min(int(k), len(cands)))
        if k == 1:
            return cands[0], {}
        if coef is None:
            scores = [float(np.ravel(c["wm_flow_resid"])[0]) for c in cands[:k]]
        else:
            scores = [float(np.ravel(c["wm_flow_resid"]) @ coef) for c in cands[:k]]
        best = int(np.argmax(scores))
        return cands[best], {"k": k, "chosen": best, "scores": np.asarray(scores, np.float32)}

    def _finish(self, pending) -> None:
        items, n, handle, t0 = pending
        try:
            N = self.num_candidates
            outs = self.policy.finish_batched(handle)[: n * N]  # row r = request r // N, candidate r % N
            dt = time.monotonic() - t0
            self.stats["requests"] += n
            self.stats["batches"] += 1
            self.stats["infer_s"] += dt
            for i, (prepped, fut) in enumerate(items):
                coef = self.rank_coef.get(prepped.get("task_id")) if prepped.get("rank_full") else None
                out, bon = self._select(outs[i * N:(i + 1) * N], prepped["num_candidates"], coef)
                a_model = np.asarray(out["actions"], np.float32)[: self.H]
                resp = {
                    "actions_model": a_model,
                    "actions_env": ar.to_env(a_model),
                    "values": {k: np.asarray(out[k], np.float32) for k in VALUE_KEYS if k in out},
                    "infer_ms": dt * 1000.0,
                    "batch": n,
                }
                if bon:
                    resp["best_of_n"] = bon
                fut.set_result(resp)
        except Exception as e:  # never kill the loop; fail the requests instead
            logger.exception("batch failed")
            for _, fut in items:
                if not fut.done():
                    fut.set_exception(e)

    # ── pipeline: the dispatcher prepares + launches batch N+1 while the GPU runs batch N ──
    def _dispatch_loop(self):
        """Transform requests as they arrive (overlapping the running batch); when the GPU frees up, launch
        everything prepared, rounded down to a compiled bucket."""
        backlog: deque = deque()

        def take(block_s):
            req, fut = self.q.get(timeout=block_s)
            try:
                obs, initial, overrides = self._prepare(req)
                prepped = self.policy.transform_one(obs, initial, overrides)
                prepped["task_id"] = int(req["task_id"])
                prepped["rank_full"] = bool(int(overrides.get("rank_full") or 0))
                backlog.append((prepped, fut))
            except Exception as e:
                logger.exception("prepare failed")
                fut.set_exception(e)

        while True:
            try:  # idle: block for work; GPU busy: poll so we notice it freeing up
                take(None if not backlog else 0.002)
            except Empty:
                pass
            if not (backlog and self._gpu_free.is_set()):
                continue
            while True:  # transform everything already queued before launching
                try:
                    take(0)
                except Empty:
                    break
            n = max(b for b in self.buckets if b <= min(len(backlog), self.max_batch))
            items = [backlog.popleft() for _ in range(n)]
            t0 = time.monotonic()
            try:
                handle = self.policy.launch([p for p, _ in items], num_candidates=self.num_candidates)
            except Exception as e:
                logger.exception("dispatch failed")
                for _, fut in items:
                    fut.set_exception(e)
                continue
            self._gpu_free.clear()
            self._inflight.put((items, n, handle, t0))

    def _finish_loop(self):
        import jax

        while True:
            pending = self._inflight.get()
            items, n, handle, t0 = pending
            try:
                jax.block_until_ready(handle["outs"])
            except Exception:
                pass  # surfaces in _finish
            self.stats["gpu_busy_s"] += time.monotonic() - max(t0, getattr(self, "_last_done", t0))
            self._last_done = time.monotonic()
            self._gpu_free.set()  # next batch launches while this one is post-processed
            self._finish(pending)

    def _infer(self, prepared: list) -> list:
        """One synchronous forward pass (warmup) on a batch of a compiled bucket size."""
        return self.policy.infer_batched(
            [p[0] for p in prepared], [p[1] for p in prepared], [p[2] for p in prepared],
            num_candidates=self.num_candidates,
        )

    def warmup(self, image_hw: int = 256) -> None:
        dummy = {
            "proprio": np.zeros(36, np.float32), "grippers": np.zeros(2, np.float32), "task_id": 0,
            "images": {cam: np.zeros((image_hw, image_hw, 3), np.uint8) for _, cam in self.data.camera_map},
            "inpaint": {"chunk": np.zeros((self.H, ar.ACTION_DIM), np.float32), "executed": 8, "keep": 4},
        }
        for b in self.buckets:
            t0 = time.monotonic()
            self._infer([self._prepare(dummy)] * b)
            t1 = time.monotonic()
            self._infer([self._prepare(dummy)] * b)
            logger.info("warmup batch %d: compile+run %.1fs, run %.0f ms", b, t1 - t0, 1000 * (time.monotonic() - t1))

    async def infer(self, req: dict) -> dict:
        fut: Future = Future()
        self.q.put((req, fut))
        return await asyncio.wrap_future(fut)


async def serve(policy: BatchedPolicy, host: str, port: int, stats_interval: float = 30.0):
    packer = msgpack_numpy.Packer()

    async def handler(ws):
        policy.clients += 1
        try:
            await _serve_client(ws)
        finally:
            policy.clients -= 1

    async def _serve_client(ws):
        async for msg in ws:
            req = msgpack_numpy.unpackb(msg)
            if req.get("type") == "ping":
                await ws.send(packer.pack({"ok": True, "metadata": policy.metadata()}))
                continue
            try:
                resp = await policy.infer(req)
            except Exception as e:
                resp = {"error": repr(e)}
            await ws.send(packer.pack(resp))

    async with ws_server.serve(handler, host, port, compression=None, max_size=None, ping_interval=None):
        logger.info("serving on %s:%d", host, port)
        last = dict(policy.stats)
        t_last = time.monotonic()
        while True:
            await asyncio.sleep(stats_interval)
            s, now = dict(policy.stats), time.monotonic()
            dr, db = s["requests"] - last["requests"], s["batches"] - last["batches"]
            if db:
                dt = now - t_last
                logger.info("clients %d requests %d batches %d | last %.0fs: %.2f req/s, mean batch %.1f, "
                            "mean latency %.0f ms, gpu busy %.0f%%", policy.clients, s["requests"], s["batches"], dt,
                            dr / dt, dr / db, 1000 * (s["infer_s"] - last["infer_s"]) / db,
                            100 * (s["gpu_busy_s"] - last["gpu_busy_s"]) / dt)
            last, t_last = s, now


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-batch", type=int, default=48)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--noise-temperature", type=float, default=1.0)
    p.add_argument("--time-threshold-inpaint", type=float, default=0.3)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--attention", choices=["vanilla", "cudnn"], default="cudnn",
                   help="cudnn: cuDNN fused attention for the VLM prefix (falls back to vanilla if unsupported)")
    p.add_argument("--buckets", default="", help="comma-separated compiled batch sizes, must include 1 "
                   "(default: 1,2,4,8,12,16,24,32,48,64 capped at --max-batch)")
    p.add_argument("--stats-interval", type=float, default=30.0)
    p.add_argument("--num-candidates", type=int, default=1,
                   help="best-of-N: flow samples per request (shared prefix); requests choose k <= N via overrides")
    a = p.parse_args()
    cfg = get_config(a.config_name)
    defaults = {"num_steps": a.num_steps, "noise_temperature": a.noise_temperature,
                "time_threshold_inpaint": a.time_threshold_inpaint, "cfg_scale": a.cfg_scale}
    import jax

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
    if a.attention == "cudnn":
        from roboeval_solution.training import cudnn_attention_patch

        cudnn_attention_patch.install()
    buckets = tuple(int(b) for b in a.buckets.split(",")) if a.buckets else None
    policy = BatchedPolicy(cfg, a.checkpoint_dir, a.max_batch, defaults,
                           buckets=buckets, num_candidates=a.num_candidates)
    asyncio.run(serve(policy, a.host, a.port, a.stats_interval))


if __name__ == "__main__":
    main()
