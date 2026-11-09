"""Parallel RoboEval rollouts against a policy server: evaluation and data collection.

Runs in the simulator venv (.venv-sim: py3.10, roboeval installed):

    MUJOCO_GL=egl .venv-sim/bin/python scripts/rollout.py --server ws://localhost:8765 \
        --out <out_dir> --episodes-per-task 20 --workers 20 --envs-per-worker 3 --gpus 0 [--record]

* Environments are built exactly like the official evaluator (roboeval.challenge.submission.make_env)
  and episodes terminate on the same conditions (success / fail / unhealthy / budget).
* Each worker process runs --envs-per-worker env slots in one single-threaded loop: while a slot
  waits for its action chunk, the others keep stepping. Slots never interleave inside an env call
  and each slot keeps its own copy of NumPy's global RNG state, so every episode is exactly what a
  dedicated process would produce.
* Jobs sit in one queue per task family; a slot stays on its family (the env is reused: make_env is
  slow) and moves to the family with the most remaining work when its queue runs dry.
* Sim speedups (roboeval_solution.sim.speedups) are bit-exact. Without --record, cameras are rendered
  only on the steps whose observation feeds a policy request (rendering does not affect the simulation).
* Per-episode official metrics -> <out>/episodes.jsonl; aggregate + RoboEval Diagnostic Score
  (official scoring code) -> <out>/summary.json.
* --record writes the episodes to one rbe store per worker under <out>/stores/w<k>/ for training.
* Error episodes count as failures (the official evaluator's convention: success 0, metrics so far).

Exit status: 0 = complete. 2 = cut short (a worker crashed or was killed, or every worker lost the
server): summary.json is still written, marked "partial": {episodes, missing, note}; every finished
episode is valid on disk and the missing ones never ran. Anything else non-zero = the run failed.

Seeds: the official evaluation seeds 1000-1049 are refused (rules forbid tuning on them).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from roboeval_solution.data.tasks import BUDGET_SECONDS, TASK_FAMILIES, TASK_TO_ID  # noqa: E402

OFFICIAL_SEEDS = set(range(1000, 1050))
log = logging.getLogger("rollout")


# ── policy client ───────────────────────────────────────────────────────────
class PolicyClient:
    """One websocket connection; requests are answered in order. A receiver thread unpacks
    responses into `inbox` as (tag, response) so a loop driving several clients never blocks."""

    def __init__(self, url: str, timeout: float = 600.0, inbox: queue.Queue | None = None, tag=None):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        self._packer = msgpack_numpy.Packer()
        self._unpack = msgpack_numpy.unpackb
        deadline = time.time() + timeout
        while True:
            try:
                self._ws = connect(url, compression=None, max_size=None, open_timeout=30)
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(2.0)
        self.metadata = self.call({"type": "ping"})["metadata"]
        self.inbox, self.tag = inbox, tag
        if inbox is not None:
            threading.Thread(target=self._receive, daemon=True).start()

    def call(self, msg: dict) -> dict:
        self._ws.send(self._packer.pack(msg))
        resp = self._unpack(self._ws.recv())
        if "error" in resp:
            raise RuntimeError(f"policy server error: {resp['error']}")
        return resp

    def send(self, msg: dict) -> None:
        self._ws.send(self._packer.pack(msg))

    def _receive(self):
        try:
            for raw in self._ws:
                self.inbox.put((self.tag, self._unpack(raw)))
        except Exception as e:  # connection closed
            self.inbox.put((self.tag, {"error": f"connection lost: {e!r}"}))

    def close(self):
        self._ws.close()


# ── environments ────────────────────────────────────────────────────────────
def make_env_any(family: str, variation: str, iface):
    """roboeval.challenge.submission.make_env (the official evaluator's)."""
    from roboeval.challenge import submission

    return submission.make_env(family, variation, iface)


def make_interface(cameras, resolution):
    from roboeval.challenge.submission import Interface

    return Interface.from_dict({
        "action_mode": {"absolute": False, "ee": True},
        "control_frequency": 20,
        "observation": {"proprioception": True, "cameras": [
            {"name": c, "resolution": [resolution, resolution], "rgb": True, "depth": False} for c in cameras]},
    })


class EnvPool:
    """Idle envs per (family, variation), bounded; slots check one out per episode."""

    def __init__(self, iface, capacity: int):
        self.iface, self.capacity = iface, capacity
        self.idle: dict = {}  # key -> [env]
        self.lru: list = []  # idle (key, env), oldest first
        self.n_alive = 0

    def get(self, key):
        envs = self.idle.get(key)
        if envs:
            env = envs.pop()
            self.lru.remove((key, env))
            return env, 0.0
        while self.n_alive >= self.capacity and self.lru:
            old_key, old = self.lru.pop(0)
            self.idle[old_key].remove(old)
            old.close()
            self.n_alive -= 1
        t = time.perf_counter()
        env = make_env_any(key[0], key[1], self.iface)
        self.n_alive += 1
        return env, time.perf_counter() - t

    def put(self, key, env):
        self.idle.setdefault(key, []).append(env)
        self.lru.append((key, env))

    def discard(self, env):
        env.close()
        self.n_alive -= 1

    def close(self):
        for _, env in self.lru:
            env.close()


# ── recording ─────────────────────────────────────────────────────────────
# Per-call head outputs, held constant over the frames executed from that call.
PRED_ARRAYS = ("pred_success", "pred_stage", "pred_togo", "pred_completion")
_PRED_SRC = {"pred_success": ("success_pred", 1), "pred_stage": ("stage_pred", 5), "pred_togo": ("togo_pred", 11),
             "pred_completion": ("completion_pred", 1)}


def _rec_ids(env):
    """Arm site ids of an env, cached on it (static after compile)."""
    ids = getattr(env, "_rbe_rec_ids", None)
    if ids is None:
        ph = env._mojo.physics
        # the IK's site names are from its own robot-only model ('\\'-namespaced); same sites in the env model use '/'
        sites = [ph.model.name2id(n.replace("\\", "/"), "site") for n in env._robot.ik_config.arm_sites]
        ids = np.asarray(sites)
        env._rbe_rec_ids = ids
    return ids


def ee_state(env) -> np.ndarray:
    """[12] measured hand poses, per arm: world position (3) + rotation vector (3) of the IK target site
    (the frame the ee-delta actions are expressed in). Read-only."""
    from scipy.spatial.transform import Rotation

    sites = _rec_ids(env)
    d = env._mojo.physics.data
    rv = Rotation.from_matrix(np.asarray(d.site_xmat[sites]).reshape(-1, 3, 3)).as_rotvec()
    return np.concatenate([np.concatenate([d.site_xpos[s], r]) for s, r in zip(sites, rv)]).astype(np.float32)


def pred_from_values(v: dict) -> dict:
    """Server `values` -> fixed-width f32 per-call predictions (NaN where the model has no such head)."""
    out = {}
    for name, (key, width) in _PRED_SRC.items():
        x = np.full(width, np.nan, np.float32)
        if key in v:
            a = np.ravel(np.asarray(v[key], np.float32))[:width]
            x[: len(a)] = a
        out[name] = x[0] if width == 1 else x
    return out


def reconcile(arrays: dict, metrics: dict, meta: dict) -> list:
    """Write-time checks: the recorded per-frame arrays reproduce the official episode metrics."""
    from roboeval_solution.data import score_reward as sr

    bad = []
    last = arrays["m_run"][-1]
    for i, k in enumerate(sr.METRIC13):
        if k != "ct" and abs(float(last[i]) - float(metrics[k])) > 1e-6 * max(1.0, abs(float(metrics[k]))):
            bad.append(f"m_run.{k}")
    acc = arrays["ev_acc"][-1]
    for k, num, den in (("mjj", 0, 2), ("mcj", 1, 2), ("bavd", 3, 4), ("bgvd", 5, 6)):
        v = acc[num] / acc[den] if acc[den] > 0 else 0.0
        if abs(v - metrics[k]) > 1e-6 * max(1.0, abs(metrics[k])):
            bad.append(f"ev_acc.{k}")
    J = meta["stage_count"]
    if abs(arrays["stage_flags"][-1, :J].sum() / J - metrics["tp"]) > 1e-6:
        bad.append("stage_flags")
    return bad


class InjectSkip(Exception):
    """An inject job (sim/inject.py) that reached no error state: no episode, the env stays usable."""


class Shift:
    """A scripted controller on a shifted clock, for recovery's track stage: it takes over at the branch frame of the
    failed episode, which matches a different frame of its reference run. A controller with a takeover(env,
    last_action) method adopts the taken-over state at its first step."""

    def __init__(self, ctl, offset: int, last_action=None):
        self.ctl, self.offset, self.last = ctl, offset, last_action

    def __call__(self, env, t: int):
        if self.last is not None:
            if hasattr(self.ctl, "takeover"):
                self.ctl.takeover(env, self.last)
            self.last = None
        return self.ctl(env, t + self.offset)


# ── episode slot ───────────────────────────────────────────────────────────
class Slot:
    """One episode at a time, advanced one env step per call so several slots share a process."""

    def __init__(self, sid: int, args, client: PolicyClient | None, record_cams: list):
        self.sid, self.args, self.client = sid, args, client
        self.cams = client.metadata["cameras"] if client is not None else []  # None: --scripted
        self.record_cams = record_cams
        self.job = None
        self.waiting = False  # request in flight
        self.np_rng_state = None
        self.family = None
        self.dead = False  # server connection lost: take no more jobs
        self.recorder = None
        if args.record:
            from roboeval_solution.sim.metric_recorder import MetricRecorder

            self.recorder = MetricRecorder()

    # NumPy's global RNG is shared by all slots of a process; the env seeds and draws from it in
    # reset() (spawn poses). Swap each slot's own state in and out around every env call.
    def _enter(self):
        if self.np_rng_state is not None:
            np.random.set_state(self.np_rng_state)

    def _exit(self):
        self.np_rng_state = np.random.get_state()

    def start(self, job, env, t_make: float):
        from roboeval_solution.sim import speedups

        a = self.args
        self.job, self.env = job, env
        self.max_steps = int(round(BUDGET_SECONDS[job["family"]] * 20))
        self.tm = {"reset": 0.0, "policy": 0.0, "env": 0.0, "encode": 0.0, "make_env": t_make}
        self.plan_model = self.plan_env = None
        self.m_run_reset = self.ev_acc_reset = None  # set by _replay on branch jobs
        self.executed = 0
        self.t = 0
        self.t_start = 0  # env clock when the policy takes over (_replay moves it past a branch prefix)
        self.frames = {k: [] for k in ("state", "state_gripper", "state_ee", "actions", "sc", "m_run", "stage_flags",
                                       *PRED_ARRAYS)}
        self.pred = None  # head outputs of the current policy call (held over the frames it covers)
        self.images = {c: [] for c in self.record_cams} if a.record else {}
        self.termination, self.n_calls, self.infer_ms = "budget", 0, 0.0
        self.info = {}
        cp = job.get("client") or {}  # per-episode client params (--param-search), else the CLI values
        self.replan = int(cp.get("replan", a.replan))
        self.keep = int(cp.get("keep", a.keep))
        self.grip_thr = float(cp.get("gripper_threshold", a.gripper_threshold))
        # playback speed: the `replan` model actions of each chunk are executed over round(replan / speed)
        # env steps (pose trajectory resampled, data/action_repr.resample_deltas)
        self.n_exec = max(1, int(round(self.replan / float(cp.get("exec_speed", a.exec_speed)))))
        # stall boost (tuning/stall.py; "stall_boost" 0 off / 1 on the P(success) trigger / 2 on any stall event):
        # the next STALL_BOOST_CALLS calls run at 1.5x noise temperature (cap 1.2), best of all server candidates.
        # Events are detected whatever the setting: the param search learns stall_boost only from stalled episodes.
        from roboeval_solution.tuning.stall import StallDetector

        self.stall_mode = int(cp.get("stall_boost") or 0)
        self.det = StallDetector(job["family"])
        self.boost_left, self.n_boost = 0, 0
        self.ctl = None
        self.aux = False  # a scripted episode that still asks the served policy for its head outputs
        if a.scripted or job.get("scripted"):  # privileged scripted policy (sim/scripted.py), one env action per step
            from roboeval_solution.sim import scripted

            # recovery's retry / track stages bring their own per-target seed (sim/recovery.script_seed)
            self.ctl, job["script_params"] = scripted.make(
                a.scripted or job["scripted"],
                seed=int(job.get("script_seed", int(job["seed"]) * 16 + int(job.get("repeat", 0)))),
                randomize=bool(job.get("script_randomize", True)), retry=bool(job.get("retry")))
            # with a server (recovery's retry / track stages): every n_exec steps it is asked only for its head
            # outputs, recorded like a rollout's (the advantages need them); its actions are not executed
            self.aux = self.client is not None and a.record
            self.aux_every, self.aux_t = self.n_exec, None
            self.replan = self.n_exec = 1
        self.t0 = time.time()
        _t = time.perf_counter()
        self._enter()
        if job.get("inject") is not None and job.get("prefix") is None:  # programmatic hard case (sim/inject.py):
            # the prefix = the scripted controller with one injected error, unrecorded, up to the error state
            from roboeval_solution.sim import inject
            from roboeval_solution.sim.recovery import RETRY_LEFT

            speedups.set_render(env, False)
            pre, info = inject.make_prefix(env, job, a.scripted or job["scripted"], self.max_steps,
                                           RETRY_LEFT[job["family"]], ee_state)
            if pre is None:
                self._exit()
                raise InjectSkip(info)
            job["prefix"], job["inject_result"] = pre, info
        if job.get("track_j") is not None:  # recovery's track stage: the reference run's controller, advanced on its
            # own episode (same seed) to the matched frame, then on its clock from there (Shift)
            speedups.set_render(env, False)
            env.reset(seed=job["seed"])
            for t in range(int(job["track_j"])):
                act = np.asarray(self.ctl(env, t), np.float32)
                act[12:14] = (act[12:14] > 0.5).astype(np.float32)
                env.step(act)
            last = job["prefix"][-1] if len(job["prefix"]) else None
            self.ctl = Shift(self.ctl, int(job["track_j"]) - len(job["prefix"]), last)
        speedups.set_render(env, True)
        self.obs, self.info = env.reset(seed=job["seed"])
        self._exit()
        self.tm["reset"] = time.perf_counter() - _t
        if self.recorder is not None:
            from roboeval_solution.data.score_reward import STAGE_COUNT
            from roboeval_solution.sim.metric_recorder import stage_row

            self.recorder.attach(env)  # each reset builds a new evaluator
            # Stage flags straight from the new evaluator: reset()'s info dict is built before the env's
            # success check runs, so it carries the previous episode's final metrics (nothing when fresh).
            flags = dict(env._evaluator._stage_flags)
            self.stage_flags_reset = stage_row({"task_stage_reached": flags})
            self.stage_count = max(flags) if flags else STAGE_COUNT[job["family"]]
        pre = job.get("prefix")
        if pre is not None and len(pre):
            self._replay(np.asarray(pre, np.float32))
        self._request()

    def _replay(self, actions: np.ndarray):
        """Branch jobs (sim/recovery.py, sim/inject.py): execute a prefix of env actions unrecorded (e.g. the original
        episode's actions, bit-exact from the same seed, plus optional backward / perturbation actions). The policy
        takes over from the resulting state and inherits the episode's clock, budget, stage flags and metric sums."""
        from roboeval_solution.sim import speedups

        _t = time.perf_counter()
        self._enter()
        for i, act in enumerate(actions):
            speedups.set_render(self.env, i == len(actions) - 1)  # images only for the first policy request
            self.obs, _, terminated, truncated, self.info = self.env.step(act)
            if self.env.success or terminated or truncated:
                self._exit()
                self.t = i + 1  # the error episode's length
                raise RuntimeError(f"branch replay ended the episode at step {i + 1}/{len(actions)}")
        self._exit()
        self.t = self.t_start = len(actions)
        self.tm["replay"] = time.perf_counter() - _t
        if self.recorder is not None:  # recorded arrays cover the policy steps only
            from roboeval_solution.sim.metric_recorder import m_run_row, stage_row

            self.ev_acc_reset = (np.asarray(self.recorder.acc[-1], np.float64) if self.recorder.acc
                                 else np.zeros(8)).tolist()  # cumulative evaluator sums at the branch
            self.m_run_reset = m_run_row(self.info, self.t - 1).tolist()  # running metrics at the branch
            self.recorder.rb.clear()
            self.recorder.acc.clear()
            self.stage_flags_reset = stage_row(self.info)

    def _request(self):
        a = self.args
        if self.ctl is not None:
            _t = time.perf_counter()
            self.plan_exec = np.asarray(self.ctl(self.env, self.t), np.float32)[None]
            self.executed = 0
            self.tm["policy"] += time.perf_counter() - _t
            if not self.aux:
                self.pred = pred_from_values({}) if a.record else None
                self.n_calls += 1
                return
            if self.aux_t is not None and self.t - self.aux_t < self.aux_every:
                return  # the held head outputs cover this frame
            self.aux_t = self.t  # ask the server for the head outputs of this frame
        req = {
            "proprio": self.obs["proprioception"], "grippers": self.obs["proprioception_grippers"],
            "images": {c: self.obs[f"rgb_{c}"] for c in self.cams}, "task_id": TASK_TO_ID[self.job["family"]],
            "episode_step": self.t,  # env clock (a branch inherits the original episode's), = the stored frame time
            "inpaint": None if self.plan_model is None or self.keep <= 0 else
            {"chunk": self.plan_model, "executed": self.replan, "keep": self.keep},
            "overrides": self._overrides(),
        }
        self._t_req = time.perf_counter()
        self.client.send(req)
        self.waiting = True

    def _overrides(self) -> dict:
        """The episode's server overrides; boosted (stall_boost) for the calls after a stall."""
        ov = self.job.get("overrides") or {}
        if self.boost_left <= 0:
            return ov
        self.boost_left -= 1
        self.n_boost += 1
        return {**ov, "noise_temperature": min(1.2, 1.5 * float(ov.get("noise_temperature") or 1.0)),
                "num_candidates": 64}  # the server takes the best of all its candidates

    def _stall_events(self, events: list) -> None:
        from roboeval_solution.tuning.stall import STALL_BOOST_CALLS

        if self.boost_left == 0 and self.det.boosts(self.stall_mode, events):
            self.boost_left = STALL_BOOST_CALLS

    def on_response(self, resp: dict):
        if "error" in resp:
            raise RuntimeError(f"policy server error: {resp['error']}")
        self.tm["policy"] += time.perf_counter() - self._t_req
        self.waiting = False
        if self.ctl is not None:  # scripted episode: the head outputs only, the scripted action stands
            self.pred = pred_from_values(resp.get("values") or {})
            self.n_calls += 1
            self.infer_ms += float(resp["infer_ms"])
            return
        from roboeval_solution.data import action_repr as ar

        self.plan_model, self.plan_env = resp["actions_model"], resp["actions_env"]
        self.plan_exec = ar.resample_deltas(self.plan_env, self.replan, self.n_exec)  # env actions of this call
        self.executed = 0
        if self.args.record:
            self.pred = pred_from_values(resp.get("values") or {})
        self.n_calls += 1
        self.infer_ms += float(resp["infer_ms"])
        self._stall_events(self.det.on_call(resp.get("values") or {}, self.t, self.boost_left > 0))

    def step(self):
        """One env step. Returns the finished episode (ep, record) or None."""
        from roboeval_solution.data.store import encode_jpeg
        from roboeval_solution.sim import speedups

        a, env, obs = self.args, self.env, self.obs
        action = np.asarray(self.plan_exec[self.executed], np.float32).copy()
        action[12:14] = (action[12:14] > self.grip_thr).astype(np.float32)
        if a.record:
            f = self.frames
            f["state"].append(np.asarray(obs["proprioception"], np.float32))
            f["state_gripper"].append(np.asarray(obs["proprioception_grippers"], np.float32))
            f["state_ee"].append(ee_state(env))
            f["actions"].append(action)
            for k in PRED_ARRAYS:
                f[k].append(self.pred[k])
            _t = time.perf_counter()
            for c in self.record_cams:
                self.images[c].append(encode_jpeg(obs[f"rgb_{c}"]))
            self.tm["encode"] += time.perf_counter() - _t
        self.executed += 1
        # images are only needed on the step whose observation feeds the next request
        speedups.set_render(env, a.record or self.executed >= self.n_exec)
        _t = time.perf_counter()
        self._enter()
        self.obs, _, terminated, truncated, self.info = env.step(action)
        success = env.success
        self._exit()
        self.tm["env"] += time.perf_counter() - _t
        info = self.info
        if a.record:  # post-step running metrics, aligned with the frame of the action
            from roboeval_solution.sim.metric_recorder import m_run_row, stage_row

            self.frames["sc"].append(int(info.get("slip_count", 0) or 0))
            self.frames["m_run"].append(m_run_row(info, self.t))
            self.frames["stage_flags"].append(stage_row(info))
        self.t += 1
        if self.ctl is None:  # stall events from what the policy observes (tuning/stall.py)
            self._stall_events(self.det.on_step(action, self.obs["proprioception_grippers"], self.t))
        done = True
        if success:
            self.termination = "success"
        elif terminated:
            self.termination = "fail"
        elif truncated:
            self.termination = "unhealthy"
        elif self.t >= self.max_steps:
            self.termination = "budget"
        else:
            done = False
        if not done:
            if self.executed >= self.n_exec:
                self._request()
            return None
        return self._finish()

    def _finish(self):
        from roboeval.challenge import results as res
        from roboeval.challenge import scoring

        steps = self.t
        raw = res.to_jsonable(self.info)
        metrics = res.episode_metrics(raw, steps, 20)
        score_ep = scoring.task_scores(self.job["family"], metrics)  # official formula on this episode
        record_out = None
        rec_meta = {}
        if self.args.record:
            arrays = {k: np.stack(v) for k, v in self.frames.items()}
            arrays["sc"] = arrays["sc"].astype(np.int32)
            arrays["rb_counts"], arrays["ev_acc"] = self.recorder.take()
            # t0: env clock at the first recorded frame (a branch's policy starts after its replayed prefix, inside
            # the same task budget); frame i was observed at env step t0 + i
            rec_meta = {"t0": int(self.t_start), "budget_steps": self.max_steps,
                        "stage_count": int(self.stage_count), "stage_flags_reset": self.stage_flags_reset.tolist()}
            if getattr(self, "m_run_reset", None) is not None:  # branched episode: running values at the branch
                rec_meta.update(m_run_reset=self.m_run_reset, ev_acc_reset=self.ev_acc_reset)
            problems = reconcile(arrays, metrics, rec_meta)
            if problems:
                log.warning("recording mismatch on %s seed %s: %s", self.job["family"], self.job["seed"], problems)
            rec_meta["recording_ok"] = not problems
            record_out = (arrays, self.images)
        job = self.job
        ep = {
            **{k: v for k, v in job.items() if k not in ("overrides", "prefix")},
            "overrides": job.get("overrides") or {},
            "termination": self.termination, "steps": steps, "success": self.termination == "success",
            "metrics": metrics, "score_ep": score_ep, "policy_calls": self.n_calls, "boosted_calls": self.n_boost,
            **({"stall": self.det.summary()} if self.ctl is None else {}),
            "infer_ms_mean": self.infer_ms / max(1, self.n_calls),
            "wall_s": time.time() - self.t0,
            "timing_s": self.tm,
            **rec_meta,
        }
        g = (self.args._score_weights or {}).get(job["family"])
        if g is not None:  # linearised score of this episode under the current weights, for monitoring
            from roboeval_solution.data import score_reward as sr

            if record_out is not None and rec_meta.get("recording_ok"):
                m = sr.step_increments(record_out[0], {**rec_meta, **ep}).sum(0)
            else:
                m = sr.fined_official(metrics, self.termination, steps, self.max_steps)
            ep["r_lin"] = float(m @ np.asarray(g))
        self.frames = self.images = None
        return ep, record_out

    def error_result(self, tb: str) -> dict:
        """The failed episode as the official evaluator counts an error: termination "error", success 0 and the
        metrics of the env info so far (none if the env never reset)."""
        ep = {**{k: v for k, v in self.job.items() if k != "prefix"}, "termination": "error", "error": tb,
              "success": False, "steps": int(self.t)}
        try:
            from roboeval.challenge import results as res
            from roboeval.challenge import scoring

            raw = res.to_jsonable(self.info or {})
            raw["success"] = 0.0
            ep["metrics"] = res.episode_metrics(raw, int(self.t), 20)
            ep["score_ep"] = scoring.task_scores(self.job["family"], ep["metrics"])
        except Exception:  # noqa: BLE001 - the error report must not fail
            pass
        return ep


def die_with_parent() -> None:
    """SIGKILL this process when its parent dies (Linux PR_SET_PDEATHSIG), so a killed coordinator does not leave
    orphaned env workers holding GBs each."""
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # 1 = PR_SET_PDEATHSIG
    except OSError:
        return
    if os.getppid() == 1:  # the parent died before the prctl
        os._exit(1)


def worker_main(wid: int, args, family_queues: dict, results: mp.Queue, home_families: list,
                param_queues: dict | None = None, no_more_jobs=None):
    die_with_parent()
    try:
        gpus = [g for g in (args.gpus or "").split(",") if g != ""]
        if gpus:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = gpus[wid % len(gpus)]
        os.environ.setdefault("MUJOCO_GL", "egl")
        logging.basicConfig(level=logging.INFO, format=f"%(asctime)s w{wid} %(message)s")
        from roboeval_solution.data.store import StoreWriter
        from roboeval_solution.sim import speedups

        args._score_weights = None
        if args.score_weights:
            sw = json.loads(Path(args.score_weights).read_text())
            args._score_weights = {f: np.asarray(g, np.float64) for f, g in sw["g"].items()}
        speedups.install_shared_gl()
        speedups.install()
        speedups.install_lazy_render()

        inbox: queue.Queue = queue.Queue()
        if args.scripted:
            clients = [None] * args.envs_per_worker
            meta = {"cameras": [], "checkpoint": f"scripted:{args.scripted}"}
        else:
            clients = [PolicyClient(args.server, inbox=inbox, tag=s) for s in range(args.envs_per_worker)]
            meta = clients[0].metadata
        record_cams = args.record_cameras if args.record else []
        cams = list(dict.fromkeys(meta["cameras"] + record_cams))
        iface = make_interface(cams, args.resolution)
        writer = None
        if args.record:
            writer = StoreWriter(Path(args.out) / "stores" / f"w{wid:02d}", cameras=args.record_cameras, fps=20,
                                 info={"source": "scripted" if args.scripted else "rollout",
                                       "server_metadata": meta, "run": args.out})
        pool = EnvPool(iface, capacity=max(args.env_cache, args.envs_per_worker))
        slots = [Slot(s, args, clients[s], record_cams) for s in range(args.envs_per_worker)]
        for s, fam in zip(slots, home_families):
            s.family = fam
        exhausted = False

        def next_job(slot):
            """From the slot's family queue; else the family with the most remaining work."""
            order = [slot.family] if slot.family else []
            sizes = {f: q.qsize() * BUDGET_SECONDS[f] for f, q in family_queues.items()}
            order += sorted((f for f in sizes if sizes[f] > 0 and f != slot.family), key=lambda f: -sizes[f])
            for fam in order:
                try:
                    job = family_queues[fam].get_nowait()
                except queue.Empty:
                    continue
                slot.family = fam
                if param_queues is not None:  # Thompson draw for this episode (pre-sampled by the coordinator)
                    draw = param_queues[fam].get(timeout=600)
                    job["ts"], job["client"] = draw["ts"], draw["client"]
                    job["overrides"] = {**(job.get("overrides") or {}), **draw["overrides"]}
                return job
            return None

        def handle(slot, outcome):
            """Store (if kept) and report a finished episode; the slot is free afterwards. Raising here leaves the
            slot's job set, so the caller's fail() reports the episode as an error."""
            ep, rec = outcome
            key = (ep["family"], ep["variation"])
            pool.put(key, slot.env)
            slot.env = None
            keep = not args.store_success_only or ep["success"]
            if keep and args.scripted and args.store_success_only:
                # scripted demos: full reward, no self-collision and no slip (env collisions are allowed: PackBox
                # spawns touching the box)
                m = ep["metrics"]
                keep = float(m["tp"]) >= 1 - 1e-6 and float(m["scc"]) == 0 and float(m["sc"]) == 0
            if keep and args.store_success_only and args.recover:  # by the target's kind (recovery.counts)
                from roboeval_solution.sim.recovery import counts
                keep = counts(ep)
            if writer is not None and rec is not None and keep:
                arrays, images = rec
                if len(arrays["actions"]) > 0:
                    ep["store"] = str(writer.root)
                    ep["store_episode_index"] = writer.add_episode(arrays, images, {
                        "family": ep["family"], "variation": ep["variation"], "seed": ep["seed"],
                        "success": ep["success"], "termination": ep["termination"],
                        "source": ep.get("source", "rollout"),
                        **{k: ep[k] for k in ("origin", "scenario", "branch_t", "script_params", "script_seed")
                           if k in ep},
                        "tp": ep["metrics"]["tp"], "job_id": ep["job_id"], "overrides": ep["overrides"],
                        "repeat": ep.get("repeat", 0),
                        "client_params": ep.get("client") or {},
                        "steps": ep["steps"], "metrics": ep["metrics"], "score_ep": ep["score_ep"],
                        "policy": meta.get("checkpoint"),
                        **{k: ep[k] for k in ("t0", "budget_steps", "stage_count", "stage_flags_reset",
                                              "recording_ok", "m_run_reset", "ev_acc_reset") if k in ep},
                    })
                    writer.flush()
            results.put(ep)
            slot.job = None

        def fail(slot, tb):
            j = slot.job or {}
            log.warning("episode %s failed:\n%s", {k: j.get(k) for k in ("family", "seed", "repeat", "scenario", "job_id")
                                                    if k in j}, tb)
            if "connection lost" in tb:
                slot.dead = True
            if slot.env is not None:
                pool.discard(slot.env)
            if slot.job is not None:  # a late response to an idle slot has no episode to report
                results.put(slot.error_result(tb))
            slot.env, slot.job, slot.waiting = None, None, False

        while True:
            # deliver responses that arrived
            while True:
                try:
                    sid, resp = inbox.get_nowait()
                except queue.Empty:
                    break
                slot = slots[sid]
                try:
                    slot.on_response(resp)
                except Exception:
                    fail(slot, traceback.format_exc(limit=6))
            progressed = False
            for slot in slots:
                if slot.job is None:
                    if exhausted or slot.dead:
                        continue
                    job = next_job(slot)
                    if job is None:
                        # dynamic job feeds (--recover) keep the queues open until the coordinator is done
                        exhausted = ((no_more_jobs is None or no_more_jobs.is_set())
                                     and all(q.qsize() == 0 for q in family_queues.values()))
                        continue
                    key = (job["family"], job["variation"])
                    slot.job, slot.t, slot.info = job, 0, {}  # what fail() reports if the start fails
                    try:
                        env, t_make = pool.get(key)
                        slot.env = env
                        slot.start(job, env, t_make)
                    except InjectSkip as e:  # no episode from this job; the env is fine
                        results.put({**{k: v for k, v in job.items() if k != "prefix"}, "termination": "skipped",
                                     "skip_reason": str(e), "success": False, "steps": 0})
                        pool.put(key, env)
                        slot.env, slot.job = None, None
                    except Exception:
                        fail(slot, traceback.format_exc(limit=6))
                    progressed = True
                elif not slot.waiting:
                    try:
                        out = slot.step()
                        if out is not None:
                            handle(slot, out)
                    except Exception:
                        fail(slot, traceback.format_exc(limit=6))
                    progressed = True
            if not progressed:
                if all(s.job is None for s in slots) and (exhausted or all(s.dead for s in slots)):
                    break
                try:
                    sid, resp = inbox.get(timeout=1.0)
                except queue.Empty:
                    continue
                slot = slots[sid]
                try:
                    slot.on_response(resp)
                except Exception:
                    fail(slot, traceback.format_exc(limit=6))
        pool.close()
        for c in clients:
            if c is not None:
                c.close()
        if writer is not None:
            writer.close()
            if writer.num_episodes == 0:  # the trainer's loader cannot read an empty store
                shutil.rmtree(writer.root, ignore_errors=True)
    except Exception:
        results.put({"worker_error": traceback.format_exc(), "worker": wid})


# ── aggregation ────────────────────────────────────────────────────────────
def summarize(episodes: list[dict]) -> dict:
    """Per-family metric means and the official scores. Error episodes count as failures, as in the official
    evaluator (success 0, the metrics reached so far; zeros if they carry none)."""
    from roboeval.challenge import scoring, spec

    per_family: dict = {}
    counts: dict = {}
    zero = {k: 0.0 for k in spec.METRIC_KEYS}
    for fam in TASK_FAMILIES:
        eps = [e for e in episodes if e["family"] == fam and ("metrics" in e or e.get("termination") == "error")]
        if not eps:
            continue
        per_family[fam] = {k: float(np.mean([e.get("metrics", zero)[k] for e in eps])) for k in spec.METRIC_KEYS}
        counts[fam] = {"episodes": len(eps), "errors": sum(e.get("termination") == "error" for e in eps)}
    scores = scoring.suite_scores(per_family)
    return {"per_family": per_family, "counts": counts, "score": scores}


def print_summary(summary: dict) -> None:
    print(f"\n{'task':24s} {'eps':>4s} {'succ':>6s} {'tp':>6s} {'ct':>6s} {'score':>6s} {'safe':>5s} {'eff':>5s} {'coord':>5s}")
    for fam, m in summary["per_family"].items():
        t = summary["score"]["tasks"][fam]
        print(f"{fam:24s} {summary['counts'][fam]['episodes']:4d} {m['success']:6.3f} {m['tp']:6.3f} {m['ct']:6.2f} "
              f"{t['score']:6.3f} {t['safety']:5.2f} {t['efficiency']:5.2f} {t['coordination']:5.2f}")
    o = summary["score"]["overall"]
    if o:
        print(f"{'OVERALL':24s} {'':4s} {np.mean([m['success'] for m in summary['per_family'].values()]):6.3f} "
              f"{np.mean([m['tp'] for m in summary['per_family'].values()]):6.3f} {'':6s} {o['score']:6.3f} "
              f"{o['safety']:5.2f} {o['efficiency']:5.2f} {o['coordination']:5.2f}")


def build_jobs(args) -> list[dict]:
    fams = TASK_FAMILIES if args.tasks == ["all"] else args.tasks
    for f in fams:
        if f not in TASK_TO_ID:
            raise SystemExit(f"unknown task {f}")
    best = {}
    if getattr(args, "best", ""):  # every task at its posterior-best params (an evaluation, not a search)
        from roboeval_solution.sim.recovery import best_params

        if not Path(args.best).exists():
            raise SystemExit(f"--best: {args.best} not found")
        best = best_params(args.best)
    jobs = []
    repeats = max(1, args.repeats_per_seed)  # rollouts from the same initial state (seed), each its own draw
    for fi, fam in enumerate(fams):
        n = -(-args.episodes_per_task // repeats)
        start = args.seed_start + fi * 100_000
        for seed in range(start, start + n):
            if seed in OFFICIAL_SEEDS:
                raise SystemExit("refusing to roll out on official evaluation seeds")
            for r in range(repeats):
                job = {"family": fam, "variation": "PositionAndOrientation", "seed": seed, "repeat": r,
                       "overrides": dict(best.get(fam, ({}, {}))[1])}
                if fam in best:
                    job["client"] = dict(best[fam][0])
                if getattr(args, "inject", 0):
                    from roboeval_solution.sim import inject

                    spec = inject.draw(fam, seed * 16 + r)
                    if spec is None:
                        continue
                    job.update(inject=spec, retry=True, scenario="inject")
                jobs.append(job)
    random.Random(0).shuffle(jobs)
    jobs.sort(key=lambda j: (j["family"], j["variation"]))
    for k, j in enumerate(jobs):
        j["job_id"] = k
    return jobs


def home_families(jobs: list[dict], n_slots: int) -> list[str]:
    """Initial family per slot, proportional to each family's total step budget."""
    work: dict = {}
    for j in jobs:
        work[j["family"]] = work.get(j["family"], 0.0) + BUDGET_SECONDS[j["family"]]
    total = sum(work.values())
    alloc = {f: w / total * n_slots for f, w in work.items()}
    n = {f: int(a) for f, a in alloc.items()}
    for f in sorted(alloc, key=lambda f: -(alloc[f] - n[f]))[: n_slots - sum(n.values())]:
        n[f] += 1
    out = [f for f in sorted(n, key=lambda f: -BUDGET_SECONDS[f]) for _ in range(n[f])]
    return out


def finalize(out: Path, param_search: str = "") -> None:
    """summary.json (+ param_search_best.json) of a run whose coordinator was killed mid-way: its finished
    episodes are complete on disk (episodes.jsonl line + store flush per episode); the rest never ran. The
    posterior (saved every 25 episodes) gives param_search_best.json if it belongs to this run."""
    eps = [json.loads(line) for line in open(out / "episodes.jsonl")]
    summary = summarize(eps)
    run_args = json.loads((out / "args.json").read_text())
    summary.update(args=run_args, env_steps=int(sum(e.get("steps", 0) for e in eps)),
                   partial={"episodes": len(eps), "note": "coordinator killed; finalised by rollout.py --finalize"})
    if param_search and Path(param_search).exists():
        from roboeval_solution.tuning.param_search import ParamSearch

        ps = ParamSearch.load(param_search)
        if ps.meta.get("run") == str(out.resolve()):
            best = {f: ps.best(f) for f in TASK_FAMILIES}
            (out / "param_search_best.json").write_text(json.dumps(
                {"n_episodes": ps.n_episodes, "policy": ps.meta.get("policy"), "best": best}, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("finalised %s: %d episodes", out, len(eps))


PS_SHARED = "state/{run}/param_search"  # bucket prefix: c<id>.json = one collector's own Thompson evidence
PS_KEEP_RUNS = 32  # published history reaches this many runs (x step_per_run steps) back (0.8^32 < 1e-3)


def ps_share(args) -> tuple[str, int] | None:
    """(run, collector id) whose Thompson evidence is shared through the bucket: the run dir of --param-search and
    the collector id from the collect dir name (c<id>_<k>_<policy>). None when the out dir has no such name."""
    m = re.match(r"c(\d+)_\d+_", Path(args.out).name)
    return (Path(args.param_search).resolve().parent.name, int(m.group(1))) if m else None


def policy_step(meta: dict) -> int | None:
    """The served policy's training step (its id step_<S> or a checkpoint path ending in <S>)."""
    m = re.search(r"(\d+)/?$", str(meta["policy_id"]))
    return int(m.group(1)) if m else None


def shared_history(history: list, share: tuple[str, int]) -> list:
    """This collector's own entries of `history` (collector `cid`, or None: recorded without sharing) + every other
    collector's published entries (bucket PS_SHARED/c<id>.json), tagged with their collector id."""
    from roboeval_solution.distributed import bucket

    run, cid = share
    own = [h for h in history if h[6] in (cid, None)]
    others, n = [], {}
    try:
        for name in bucket.list_dir(PS_SHARED.format(run=run)):
            m = re.fullmatch(r"c(\d+)\.json", name)
            if not m or int(m.group(1)) == cid:
                continue
            d = json.loads(bucket.get_text(f"{PS_SHARED.format(run=run)}/{name}") or "{}")
            # another collector's run index means nothing here: only entries with a policy step count
            rows = [list(h[:5]) + [h[5], int(m.group(1))] for h in d.get("history", []) if h[5] is not None]
            others += rows
            n[int(m.group(1))] = len(rows)
    except Exception as e:  # no bucket / no network: this collector's own evidence only
        log.warning("param search: the other collectors' evidence not read (%s): own history only", e)
    if n:
        log.info("param search: + other collectors' evidence %s", n)
    return own + others


def publish_history(ps, share: tuple[str, int], step_per_run: float) -> None:
    """This collector's own recent Thompson evidence to the bucket (PS_SHARED/c<id>.json) for the others' refits."""
    from roboeval_solution.distributed import bucket

    run, cid = share
    cur_step, cur_run = ps.meta.get("step"), int(ps.meta.get("run_index", 0))
    keep = []
    for h in ps.history:  # own entries with a policy step (the others age them by step), at most PS_KEEP_RUNS back
        if h[6] not in (cid, None) or h[5] is None:
            continue
        age = (cur_step - h[5]) / step_per_run
        if age <= PS_KEEP_RUNS:
            keep.append(list(h[:5]) + [h[5], cid])
    try:
        bucket.put_text(f"{PS_SHARED.format(run=run)}/c{cid}.json",
                        json.dumps({"collector": cid, "step": cur_step, "run_index": cur_run, "history": keep}))
        log.info("param search: published %d own episodes (collector %d)", len(keep), cid)
    except Exception as e:  # the next collect publishes again
        log.warning("param search: publishing the evidence failed (%s)", e)


class ParamTuner:
    """Coordinator side of --param-search: Thompson draws per family, pre-sampled into small queues the
    workers take from at episode start (a draw is at most --ps-buffer episodes behind the posterior),
    and a posterior update from each finished episode's official per-episode task score."""

    def __init__(self, args, ctx, families: list, jobs: list):
        from roboeval_solution.tuning.param_search import DEFAULT_SPACE, ParamSearch

        client = PolicyClient(args.server)
        meta = client.metadata
        client.close()
        allowed = set(meta["per_request_overrides"])
        space = [p for p in DEFAULT_SPACE if p.where == "client" or p.name in allowed]
        max_cands = int(meta["max_candidates"])  # the server's --num-candidates N caps the best-of-k
        space = [dataclasses.replace(p, hi=float(min(p.hi, max_cands))) if p.name == "num_candidates" else p
                 for p in space]
        self.path = Path(args.param_search)
        run = str(Path(args.out).resolve())
        # every collector's evidence counts: each publishes its own after a collect, each refit merges the others';
        # recency by policy step (comparable across collectors)
        self.share, self.step_per_run = ps_share(args), float(args.ps_step_per_run)
        step = policy_step(meta)
        self.ps = None
        old = ParamSearch.load(self.path, seed=args.seed_start) if self.path.exists() else None
        if old is None and self.share:  # a new collector: start from the others' evidence
            old = ParamSearch(space, TASK_FAMILIES, seed=args.seed_start)
            old.meta["run_index"] = -1
        if old is not None:
            new_run = old.meta.get("run") != run  # same run = a restart: keep the evidence as it is
            if new_run or [dataclasses.asdict(p) for p in old.space] != [dataclasses.asdict(p) for p in space]:
                # a new run (or a changed space) refits on the history with recency weights: ps_decay per
                # step_per_run policy steps of age; the old state stays as .bak
                if self.path.exists():
                    shutil.copy2(self.path, self.path.with_name(self.path.name + ".bak"))
                if self.share:
                    old.history = shared_history(old.history, self.share)
                idx = int(old.meta.get("run_index", 0)) + int(new_run)
                self.ps = ParamSearch.refit(old, space, TASK_FAMILIES, seed=args.seed_start, decay=args.ps_decay,
                                            run_index=idx, step=step, step_per_run=self.step_per_run)
                log.info("param search run %d (policy step %s): refit on %d of %d history episodes (x%.2f per %g "
                         "steps, effective %.0f)", idx, step, len(self.ps.history), len(old.history), args.ps_decay,
                         self.step_per_run, self.ps.effective)
            else:
                self.ps = old
        if self.ps is None:
            self.ps = ParamSearch(space, TASK_FAMILIES, seed=args.seed_start)
        self.ps.meta.update(step=step, collector=self.share[1] if self.share else None)
        self.ps.meta["run"] = run
        self.ps.meta["policy"] = meta["policy_id"]
        self.ps.meta["history_at_start"] = len(self.ps.history)
        self.ps.save(self.path)  # the refit is on disk before the first episode
        self.server_keys = {p.name for p in space if p.where == "server"}
        self.buffer = args.ps_buffer
        self.queues = {f: ctx.Queue() for f in families}
        self.left = {f: sum(j["family"] == f for j in jobs) for f in families}
        self.n_new = 0
        log.info("param search over %s (%d prior episodes)", [p.name for p in space], self.ps.n_episodes)

    def draw(self, fam: str) -> dict:
        cfg = self.ps.sample(fam)
        return {"ts": cfg, "overrides": {k: v for k, v in cfg.items() if k in self.server_keys and v is not None},
                "client": {p.name: cfg[p.name] for p in self.ps.space if p.where == "client"}}

    def top_up(self):
        for fam, q in self.queues.items():
            if self.left[fam] > 0:
                for _ in range(max(0, self.buffer - q.qsize())):
                    q.put(self.draw(fam))

    def update(self, ep: dict):
        self.left[ep["family"]] -= 1
        if "ts" in ep and "score_ep" in ep and ep.get("termination") != "error":  # error episodes are no evidence
            # stall_boost is evidence only where a stall was detected (tuning/stall.py; ParamSearch GATED)
            ts = {**ep["ts"], "stall_seen": bool((ep.get("stall") or {}).get("first"))}
            self.ps.update(ep["family"], ts, reward=float(ep["score_ep"]["score"]))
            self.n_new += 1
            if self.n_new % 25 == 0:
                self.ps.save(self.path)

    def close(self, out: Path):
        self.ps.save(self.path)
        if self.share:
            publish_history(self.ps, self.share, self.step_per_run)
        best = {f: self.ps.best(f) for f in TASK_FAMILIES}
        (out / "param_search_best.json").write_text(json.dumps(
            {"n_episodes": self.ps.n_episodes, "policy": self.ps.meta.get("policy"), "best": best}, indent=1))
        for f, cfg in best.items():
            log.info("best %-24s %s", f, {k: round(v, 3) if isinstance(v, float) else v for k, v in cfg.items()})


# recovery stages (sim/recovery.py) in order; each "|"-separated group is a chain of its own
RECOVERY_CASCADE = "drop,perturb,reverse,peak,back2s|retry"


class RecoveryCascade:
    """--recover: for each recovery target of a collect (sim/recovery.py: failures, and successes that missed a
    stage), its applicable stages in cascade order, one try each. The next stage is queued only if the previous try
    did not count (recovery.counts); the first that counts ends the chain. The groups of RECOVERY_CASCADE are
    independent chains per target (a target is recovered if any of its chains is)."""

    def __init__(self, args):
        from roboeval_solution.sim import recovery as rec

        self.rec = rec
        rng = np.random.default_rng(args.recover_seed)
        groups = [[s for s in g.split(",") if s] for g in RECOVERY_CASCADE.split("|")]
        self.stages = [s for g in groups for s in g]
        fails = rec.find_failures(args.recover)
        fails = [fails[i] for i in rng.permutation(len(fails))]
        if args.recover_max:
            fails = fails[: args.recover_max]
        self.chains = [(fi, g) for fi in range(len(fails)) for g in groups if g]  # (target, its stages)
        if args.recover_track:  # every failed episode with a deviation onset: a track chain of its own
            from roboeval_solution.sim import deviation

            t0 = time.time()
            cand = rec.failed_episodes(args.recover)
            found = deviation.find_track(cand, workers=args.recover_track)
            log.info("track: deviation onsets in %d / %d failed episodes %s (%.0f s)", len(found), len(cand),
                     deviation.summary(found, cand), time.time() - t0)
            for i in sorted(found):
                f = cand[i]
                f["track"] = found[i]
                f["origin"].update(t_onset=found[i]["t_b"], what=found[i]["what"], ref_k=found[i]["ref_k"],
                                   ref_j=found[i]["j"])
                self.chains.append((len(fails), ["track"]))
                fails.append(f)
            if "track" not in self.stages:
                self.stages.append("track")
        self.fails, self.rng = fails, rng
        self.params = rec.best_params(Path(args.recover) / "param_search_best.json")
        self.pos = [0] * len(self.chains)  # next stage index per chain
        self.state = ["open"] * len(self.chains)  # open / recovered / exhausted
        self.tried = {s: [0, 0] for s in self.stages}  # stage -> [tries, successes]
        self.tries = [[] for _ in fails]  # per target: (stage, success, tp) of each try
        self.n_job, self.n_done, self.last_log = 0, 0, 0
        det = {}
        for f in fails:
            k = f"{f['origin']['target']}: {f.get('detector')}"
            det[k] = det.get(k, 0) + 1
        log.info("recovery over %s: %d targets %s (%d dropped the object), cascade %s", args.recover, len(fails),
                 det, sum(f["dropped"] for f in fails), self.stages)

    def _next(self, ci: int):
        fi, stages = self.chains[ci]
        f = self.fails[fi]
        while self.pos[ci] < len(stages):
            st = stages[self.pos[ci]]
            self.pos[ci] += 1
            made = self.rec.stage_prefix(f, st, self.rng)
            if made is not None:
                job = self.rec.make_job(f, st, made[0], made[1], self.params, chain=ci, attempt=self.pos[ci],
                                        repeat=self.pos[ci] - 1, job_id=self.n_job)
                self.n_job += 1
                return job
        self.state[ci] = "exhausted"
        self.n_done += 1
        return None

    def initial_jobs(self) -> list:
        jobs = [self._next(ci) for ci in range(len(self.chains))]
        return [j for j in jobs if j is not None]

    def result(self, ep: dict):
        ci = ep.get("chain")
        if ci is None:
            return None
        self.tried[ep["scenario"]][0] += 1
        self.tries[self.chains[ci][0]].append((ep["scenario"], bool(ep.get("success")),
                               float((ep.get("metrics") or {}).get("tp", ep.get("tp", 0.0)) or 0.0)))
        if self.rec.counts(ep):
            self.tried[ep["scenario"]][1] += 1
            self.state[ci] = "recovered"
            self.n_done += 1
            return None
        return self._next(ci)

    def recovered(self) -> list:
        """Per target: did any of its chains recover it."""
        out = [False] * len(self.fails)
        for (fi, _), st in zip(self.chains, self.state):
            out[fi] |= st == "recovered"
        return out

    def progress_tick(self) -> bool:
        step = max(1, len(self.chains) // 20)
        if self.n_done - self.last_log >= step or self.n_done == len(self.chains):
            self.last_log = self.n_done
            return True
        return False

    def status(self) -> str:
        rec = sum(self.recovered())
        return (f"{self.n_done}/{len(self.chains)} chains done, {rec}/{len(self.fails)} targets recovered "
                + " ".join(f"{s} {v[1]}/{v[0]}" for s, v in self.tried.items()))

    @staticmethod
    def _episode(f) -> tuple:
        o = f["origin"]
        return o.get("collect"), o.get("store"), o.get("episode")

    def episodes_recovered(self) -> dict:
        """Per original episode (a track chain is a second target of the same failed episode): recovered by any."""
        out = {}
        for f, rec in zip(self.fails, self.recovered()):
            out[self._episode(f)] = out.get(self._episode(f), False) or rec
        return out

    def left(self) -> list:
        """The episodes no chain recovered: seed, origin and every try."""
        done, out = self.episodes_recovered(), {}
        for fi, f in enumerate(self.fails):  # one entry per episode, the tries of all its chains
            if done[self._episode(f)]:
                continue
            e = out.setdefault(self._episode(f), {"family": f["family"], "variation": f["variation"], "seed": f["seed"],
                                                  "detector": f.get("detector"), "dropped": f["dropped"],
                                                  "origin": f["origin"], "tries": []})
            e["tries"] += [{"stage": s, "success": ok, "tp": tp} for s, ok, tp in self.tries[fi]]
        return list(out.values())

    def summary(self) -> dict:
        by_det = {}
        ok = self.recovered()
        for f, r in zip(self.fails, ok):
            d = by_det.setdefault(f"{f['origin']['target']}: {f.get('detector')}",
                                  {"targets": 0, "recovered": 0})
            d["targets"] += 1
            d["recovered"] += r
        eps = self.episodes_recovered()
        return {"failures": len(self.fails), "recovered": sum(ok), "episodes": len(eps),
                "episodes_recovered": sum(eps.values()), "by_detector": by_det,
                "exhausted": len(ok) - sum(ok), "dropped_object": sum(f["dropped"] for f in self.fails),
                "per_stage": {s: {"tries": v[0], "successes": v[1]} for s, v in self.tried.items()},
                "cascade": self.stages}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="ws://localhost:8765")
    p.add_argument("--out", default="")
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=100_000)
    p.add_argument("--repeats-per-seed", type=int, default=1,
                   help="episodes per initial state: --episodes-per-task / R seeds per task, each rolled out R times")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--envs-per-worker", type=int, default=3,
                   help="env slots stepped by one process while the others wait for the policy")
    p.add_argument("--gpus", default="0", help="EGL render devices, round-robin over workers")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--replan", type=int, default=8, help="actions executed per policy call")
    p.add_argument("--keep", type=int, default=4, help="inpainted actions carried into the next chunk")
    p.add_argument("--exec-speed", type=float, default=1.0,
                   help="playback speed: the --replan actions of a chunk run over round(replan / speed) env steps")
    p.add_argument("--gripper-threshold", type=float, default=0.5, help="gripper command binarisation threshold")
    p.add_argument("--param-search", default="",
                   help="Thompson-sampling state file (created if missing): every episode draws its inference / "
                        "client params from the posterior and updates it with its task score")
    p.add_argument("--best", default="", help="param_search_best.json: every task at its posterior-best client "
                   "params and server overrides (an evaluation of fixed settings; do not combine with --param-search)")
    p.add_argument("--ps-buffer", type=int, default=2, help="pre-sampled draws queued per family")
    p.add_argument("--ps-step-per-run", type=float, default=500.0,
                   help="policy training steps per --ps-decay factor (about one collect cycle): the age unit of an "
                        "episode in the refit, comparable across collectors")
    p.add_argument("--ps-decay", type=float, default=0.8,
                   help="Thompson recency weight: at each new run the posterior is refit on the history with an "
                        "episode weighing ps_decay^age (age in --ps-step-per-run policy steps)")
    p.add_argument("--env-cache", type=int, default=4, help="envs kept alive per worker (>= envs-per-worker)")
    p.add_argument("--record", action="store_true")
    p.add_argument("--store-success-only", action="store_true", help="write only successful episodes to the stores "
                   "(stricter with --scripted, see there; with --recover only complete ones)")
    p.add_argument("--recover", default="", help="collect dir: recovery cascade over its failures and successes "
                   "that missed a stage (sim/recovery.py): per target the stages are tried in order, the next one only "
                   "if the previous try was not complete; runs with the collect's task-best params (not with "
                   "--param-search)")
    p.add_argument("--recover-max", type=int, default=0, help="targets to try (random sample; 0 = all)")
    p.add_argument("--inject", type=int, default=0,
                   help="with --scripted: programmatic hard cases (sim/inject.py): each episode = the scripted policy "
                        "with one injected execution error, unrecorded up to the error state, then recorded from there "
                        "(recovery's Retry); jobs that reach no error state are skipped")
    p.add_argument("--recover-track", type=int, default=0,
                   help="track stage (sim/deviation.py): process-pool size of the onset search over every failed "
                        "episode (CPU, before the jobs start); each one with an onset gets a track chain; 0 = off")
    p.add_argument("--recover-seed", type=int, default=0)
    p.add_argument("--finalize", default="", help="run dir whose coordinator was killed: write its summary.json "
                   "(+ param_search_best.json from --param-search) from the finished episodes, then exit")
    p.add_argument("--recover-policy", default="", help="label of the served policy, for recovery.json")
    p.add_argument("--score-weights", default="",
                   help="JSON file {'g': {family: g[13]}}: adds the linearised score r_lin to each episode")
    p.add_argument("--record-cameras", nargs="+", default=["head", "front", "external", "left_wrist", "right_wrist"])
    p.add_argument("--scripted", default="", help="act with a privileged scripted policy instead of the server "
                   "(sim/scripted.py POLICIES: packbox, stack, ...); episodes are tagged source 'scripted'; with "
                   "--store-success-only only full-reward episodes without self-collision or slip are stored")
    args = p.parse_args()
    if not args.out and not args.finalize:
        p.error("--out is required")
    if args.recover and args.param_search:  # recovery tries are no evidence for the posterior, and use fixed params
        p.error("--recover runs with the collect's best params: drop --param-search")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.finalize:
        finalize(Path(args.finalize), args.param_search)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    cascade = None
    if args.recover:
        from roboeval_solution.sim import recovery as rec

        # pairs the run with its collect in training, whatever the run dir is called
        (out / rec.META).write_text(json.dumps({"collect": Path(args.recover).resolve().name,
                                                "policy": args.recover_policy,
                                                "cascade": RECOVERY_CASCADE.replace("|", ",").split(","),
                                                "seed": args.recover_seed}))
        cascade = RecoveryCascade(args)
        jobs = cascade.initial_jobs()
    else:
        jobs = build_jobs(args)
    if args.scripted:
        from roboeval_solution.sim import scripted

        fam = scripted.family(args.scripted)
        if any(j["family"] != fam for j in jobs):
            raise SystemExit(f"--scripted {args.scripted} only acts in {fam} (use --tasks {fam})")
    for j in jobs:
        if j["seed"] in OFFICIAL_SEEDS:
            raise SystemExit("refusing to roll out on official evaluation seeds")
        if args.scripted:
            j["source"] = "scripted"
    if not jobs:
        log.warning("no jobs")
        (out / "summary.json").write_text(json.dumps({"episodes": 0, "args": vars(args)}, indent=1))
        return
    n_slots = args.workers * args.envs_per_worker
    log.info("%d jobs over %d workers x %d env slots", len(jobs), args.workers, args.envs_per_worker)
    ctx = mp.get_context("spawn")
    family_queues = {f: ctx.Queue() for f in TASK_FAMILIES}
    rq = ctx.Queue()
    for j in jobs:
        family_queues[j["family"]].put(j)
    homes = home_families(jobs, n_slots)
    # interleave so every worker starts on a mix of families
    per_worker = [[homes[(s * args.workers + w) % len(homes)] for s in range(args.envs_per_worker)]
                  for w in range(args.workers)]
    tuner = None
    if args.param_search:
        tuner = ParamTuner(args, ctx, list(family_queues), jobs)
        tuner.top_up()
    time.sleep(0.5)  # let the feeder threads flush the queues (qsize is used for scheduling)
    no_more_jobs = ctx.Event()
    if cascade is None:
        no_more_jobs.set()  # a fixed job list: workers stop when the queues run dry
    procs = [ctx.Process(target=worker_main, args=(w, args, family_queues, rq, per_worker[w],
                                                   tuner.queues if tuner else None, no_more_jobs), daemon=False)
             for w in range(args.workers)]
    for pr in procs:
        pr.start()
    episodes = []
    t0 = time.time()
    steps_done = 0
    pending = len(jobs)  # queued or running
    lost = []  # workers that crashed or were killed: their running episodes never report

    def worker_lost(what: str):
        # the lost episodes keep `pending` above 0, so stop waiting for them: no new jobs once the queues are
        # empty (a cascade would otherwise keep its queues open forever); the other workers finish and exit
        lost.append(what)
        no_more_jobs.set()

    with open(out / "episodes.jsonl", "a") as f:
        while pending > 0:
            if tuner is not None:
                tuner.top_up()
            try:
                r = rq.get(timeout=0.1 if tuner is not None else 5.0)
            except queue.Empty:
                if not any(pr.is_alive() for pr in procs):
                    log.error("all workers exited with %d/%d episodes", len(episodes), len(jobs))
                    break
                for w, pr in enumerate(procs):
                    if pr.exitcode not in (None, 0) and f"w{w}" not in lost:
                        log.error("worker %d died (exit code %s): its running episodes are lost", w, pr.exitcode)
                        worker_lost(f"w{w}")
                continue
            if "worker_error" in r:
                log.error("worker %s crashed:\n%s", r["worker"], r["worker_error"])
                worker_lost(f"w{r['worker']}")
                continue
            episodes.append(r)
            pending -= 1
            if cascade is not None:  # next stage of this failure's cascade, unless it just recovered
                nj = cascade.result(r)
                if nj is not None:
                    family_queues[nj["family"]].put(nj)
                    pending += 1
                if pending == 0:
                    no_more_jobs.set()
            if tuner is not None:
                tuner.update(r)
            steps_done += int(r.get("steps", 0))
            f.write(json.dumps(r) + "\n")
            f.flush()
            done = len(episodes)
            if cascade is not None:
                if cascade.progress_tick():
                    log.info("recovery: %s, %.1f eps/min", cascade.status(), done / max(1e-6, time.time() - t0) * 60)
            elif done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                sr = np.mean([e["success"] for e in episodes])
                el = max(1e-6, time.time() - t0)
                log.info("%d/%d episodes, success %.3f, %.1f eps/min, %.0f env steps/s", done, len(jobs), sr,
                         done / el * 60, steps_done / el)
    for pr in procs:
        pr.join(timeout=60)
    if tuner is not None:
        tuner.close(out)
    no_more_jobs.set()
    summary = summarize(episodes)
    if cascade is not None:
        summary["recovery"] = cascade.summary()
        left = cascade.left()
        with open(out / "left.jsonl", "w") as f:
            for x in left:
                f.write(json.dumps(x) + "\n")
        summary["recovery"]["left"] = {fam: sum(x["family"] == fam for x in left)
                                       for fam in sorted({x["family"] for x in left})}
        log.info("recovery done: %s; %d left (left.jsonl)", cascade.status(), len(left))
    summary["wall_s"] = time.time() - t0
    summary["env_steps"] = steps_done
    summary["args"] = vars(args)
    if pending > 0:  # cut short: the finished episodes are complete on disk, the rest never ran
        summary["partial"] = {"episodes": len(episodes), "missing": pending,
                              "note": f"cut short (workers lost: {lost or 'all exited'})"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    stores = out / "stores"
    if stores.is_dir() and not any(stores.iterdir()):  # nothing kept: a run without a stores dir is not training
        stores.rmdir()                                   # data (flywheel.scripted_runs)
    print_summary(summary)
    if pending > 0:
        log.error("run cut short: %d episode(s) missing; summary.json is marked partial (exit 2)", pending)
        sys.exit(2)


if __name__ == "__main__":
    main()
