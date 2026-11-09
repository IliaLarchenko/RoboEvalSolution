"""RL flywheel driver, one process per machine; scripts/bucket_daemon.py moves the data between them.

  trainer    one long train.py run to --total-steps on the demos, the scripted generations and the newest collects
             with their recovery runs (training/flywheel.py: score-exact advantages, LeHome share decay); train.py
             swaps new data in-process (flywheel.DataRefresher) and exports + pushes a policy at every save
             (scripts/bucket_sync.py export-policy -> bucket policies/<name>/step_S, LATEST).
  collector  forever, each job on the newest pulled policy export: the previous collect's recovery cascade
             (rollout.py --recover -> runs/<name>/recovery/r<id>_<k>_<policy>/), the between-collects command
             (scripts/scripted_round.sh: one scripted generation; no policy server, one GPU job at a time), then
             collect k (rollout.py --record -> runs/<name>/rollouts/c<id>_<k>_<policy>/ + READY, pushed by the daemon).

    .venv/bin/python scripts/rl_loop.py --config configs/trainer.yaml --name <name>
    .venv/bin/python scripts/rl_loop.py --config configs/collector.yaml --name <name>
    (the configs hold the final run's settings; any flag given on the command line overrides them)

State is kept in runs/<name>/state.json, so either role can be stopped and restarted.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pickle
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SOL = Path(__file__).resolve().parents[1]
ROOT = SOL.parent
SIM_PY = SOL / ".venv-sim" / "bin" / "python"
TRAIN_PY = SOL / ".venv" / "bin" / "python"
NVCC_BIN = SOL / ".venv/lib/python3.12/site-packages/nvidia/cu13/bin"

log = logging.getLogger("rl_loop")


class State:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.d = json.loads(path.read_text()) if path.exists() else {"exports": [], "collected": []}

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d, indent=1))
            os.replace(tmp, self.path)


def log_handlers(path: Path) -> list:
    """stderr and the log file, or only stderr when stderr already appends to that file (otherwise every line
    would be written twice)."""
    try:
        same = path.exists() and os.path.samestat(os.fstat(2), path.stat())
    except OSError:
        same = False
    return [logging.StreamHandler()] if same else [logging.StreamHandler(), logging.FileHandler(path)]


def taskset(cpus: str) -> list:
    """CPU pinning prefix: the policy server's launch / XLA host threads must not wait behind the env workers'
    CPU load, so the server and the workers get disjoint cores."""
    return ["taskset", "-c", cpus] if cpus else []


def gpu_env(mem_fraction: float) -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("RBE_TRAIN_GPU", "0")
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    env["PATH"] = f"{NVCC_BIN}:{env['PATH']}"
    return env


# ── trainer ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def refresh_config(a):
    """flywheel.DataRefreshConfig from the CLI (share decay + in-process pool refresh settings)."""
    from roboeval_solution.training.flywheel import DataRefreshConfig

    return DataRefreshConfig(run_dir=str(Path(a.runs_dir) / a.name), pool_window=a.pool_window,
                             steps_per_decay=a.steps_per_round,
                             demo_weight=a.demo_weight, demo_advantage=a.demo_advantage,
                             rollout_decay=a.rollout_decay,
                             python=str(TRAIN_PY), recovery_window=a.recovery_window,
                             scripted_window=a.scripted_window)


def build_train_config(a, pool: list[str], rec: list[str], target_step: int, resume: bool, step: int = 0):
    """`pool` / `rec`: the collects and recovery runs whose advantages were just computed (one listing for both)."""
    from roboeval_solution.training import config as _config
    from roboeval_solution.training import flywheel

    cfg = _config.get_config(a.base_config)
    sources = flywheel.build_sources(cfg.data.sources, refresh_config(a), pool, step, rec=rec)
    data = dataclasses.replace(cfg.data, sources=tuple(sources))
    kw = dict(exp_name=a.name, num_train_steps=target_step, resume=resume, data=data,
              save_interval=a.save_interval, data_refresh=refresh_config(a),
              lr_schedule=dataclasses.replace(cfg.lr_schedule, decay_steps=a.total_steps,
                                              **({"peak_lr": a.peak_lr} if a.peak_lr else {}),
                                              **({"decay_lr": a.decay_lr} if a.decay_lr else {}),
                                              cooldown_start=a.cooldown_start or None,
                                              cooldown_steps=a.cooldown_steps, cooldown_lr=a.cooldown_lr))
    if a.batch_size:
        kw["batch_size"] = a.batch_size
    if a.num_workers:
        kw["num_workers"] = a.num_workers
    return dataclasses.replace(cfg, **kw)


def latest_step(ckpt_dir: Path) -> int:
    steps = [int(p.name) for p in ckpt_dir.glob("[0-9]*") if p.is_dir() and (p / "params").exists()]
    return max(steps) if steps else 0


def train_round(a, state: State, rnd: int) -> Path | None:
    """One train.py run from the newest checkpoint (or the init) to --stop-step (default --total-steps); new data
    comes in through the in-process refresh. Exports the last step. None when training is already there."""
    from roboeval_solution.training import config as _config
    from roboeval_solution.training import flywheel

    run_dir = Path(a.runs_dir) / a.name
    ckpt_dir = Path(_config.get_config(a.base_config).checkpoint_base_dir).resolve() / a.base_config / a.name
    cur, stop = latest_step(ckpt_dir), a.stop_step or a.total_steps
    if cur >= stop:
        log.info("training is at step %d (stop at %d): done", cur, stop)
        return None
    pool = flywheel.current_pool(run_dir, a.pool_window)
    rec = flywheel.recovery_runs(run_dir, a.recovery_window)
    if pool:
        flywheel.compute_advantages(refresh_config(a), pool, rnd, rec)
    cfg = build_train_config(a, pool, rec, stop + 1, resume=cur > 0, step=cur)
    cfg_path = run_dir / "configs" / f"train_r{rnd:03d}.pkl"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_bytes(pickle.dumps(cfg))
    log.info("round %d: training %d -> %d on %d sources (pool %d collects, %d recovery runs)", rnd, cur,
             stop, len(cfg.data.sources), len(pool), len(rec))
    with open(run_dir / "logs" / f"train_r{rnd:03d}.log", "w") as f:
        r = subprocess.run([str(TRAIN_PY), "scripts/train.py", "--config_file", str(cfg_path)], cwd=SOL,
                           env=gpu_env(a.train_mem), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"training round {rnd} failed (see {run_dir}/logs/train_r{rnd:03d}.log)")
    step = latest_step(ckpt_dir)
    export = run_dir / "policies" / f"step_{step:06d}"
    if not export.exists():  # bf16 params + assets, on CPU; bucket_daemon.py pushes it
        subprocess.run([str(TRAIN_PY), "-c", "import sys; from roboeval_solution.distributed.export import "
                        "export_policy; export_policy(sys.argv[1], sys.argv[2])", str(ckpt_dir / str(step)), str(export)],
                       cwd=SOL, env=dict(os.environ, JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES=""), check=True)
        if (run_dir / "score_weights.json").exists():  # the current score weights ship with the policy
            shutil.copy(run_dir / "score_weights.json", export / "score_weights.json")
    with state.lock:
        state.d["exports"].append({"round": rnd, "step": step, "path": str(export), "time": time.time()})
    state.save()
    return export


# ── collector ────────────────────────────────────────────────────────────────────────────────────────────────────────
class Server:
    def __init__(self, a, policy: Path, log_path: Path):
        self.port = a.port
        self.log_path = log_path
        self.f = open(log_path, "w")
        self.p = subprocess.Popen(
            taskset(a.server_cpus) + [str(TRAIN_PY), "scripts/serve_policy.py", "--config-name", a.base_config, "--checkpoint-dir", str(policy),
             "--port", str(a.port), "--max-batch", str(a.max_batch), *a.serve_args.split()],
            cwd=SOL, env=gpu_env(a.serve_mem), stdout=self.f, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.time() + 1800
        while "serving on" not in log_path.read_text():
            if self.p.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"policy server failed to start, see {log_path}")
            time.sleep(3)

    def stop(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)
        self.f.close()


def newest_export(pdir: Path) -> Path | None:
    """The newest complete policy export (step_* with params) under runs/<run>/policies, or None."""
    exports = sorted(p for p in pdir.glob("step_*") if p.is_dir() and not p.name.endswith(".tmp")
                     and (p / "params").exists())
    return exports[-1] if exports else None


def collect_seed_start(a, k: int) -> int:
    """--seed-start of collect k (rollout.py: task index i gets seeds start + i * 100000 + 0..)."""
    return a.seed_base + a.collector_id * 50_000_000 + k * 1000


def between_round(a, state: State, run_dir: Path, k: int, env: dict) -> None:
    """The between-collects command (e.g. scripts/scripted_round.sh: one scripted generation), once per collect number:
    a restart after it ran goes straight to the collect instead of starting another generation. An incomplete round
    is continued by the next cycle's."""
    if state.d.get("between_done") == k:
        log.info("between-collects command before collect %d already ran: skipped", k)
        return
    t0 = time.time()
    path = run_dir / "logs" / f"between_c{k:03d}.log"
    with open(path, "a") as f:
        rb = subprocess.run(["bash", "-c", a.between_cmd], cwd=SOL, env=env, stdout=f, stderr=subprocess.STDOUT)
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    (log.info if rb.returncode == 0 else log.error)(
        "between-collects command before collect %d: exit %d in %.0fs%s", k, rb.returncode, time.time() - t0,
        f": {lines[-1]}" if lines else "")
    with state.lock:
        state.d["between_done"] = k
    state.save()


def wait_for_disk(a, run_dir: Path) -> None:
    """A collect writes tens of GB while the previous one stays on disk until this one is pushed: wait (loudly) for
    --min-free-gb rather than let the rollout die on a full disk."""
    n = 0
    while a.min_free_gb > 0:
        free = shutil.disk_usage(run_dir).free / 2**30
        if free >= a.min_free_gb:
            return
        if n % 10 == 0:
            log.error("only %.0f GB free (--min-free-gb %d for a collect): waiting; free space under %s "
                      "(pushed local copies, recovery stores)", free, a.min_free_gb, run_dir)
        n += 1
        time.sleep(60)


def collect(a, state: State, pick, k: int) -> Path:
    """One collector cycle: the previous collect's recovery cascade, the between-collects command (no server: one GPU
    job at a time), then collect k. `pick()` returns the newest policy export: every collection job (recovery,
    collect) starts on the newest export at that moment."""
    run_dir = Path(a.runs_dir) / a.name
    env = dict(os.environ, MUJOCO_GL="egl")
    prev = pending_recovery(state)  # the previous collect's hard cases first: the newest policy recovers best
    if prev is not None:
        rpol = pick()
        server = Server(a, rpol, run_dir / "logs" / f"serve_rec_c{k:03d}.log")
        try:
            recover(a, state, run_dir, Path(prev["run"]), int(prev["collect"]), rpol, env)
        finally:
            server.stop()
    if a.between_cmd:
        between_round(a, state, run_dir, k, env)
    wait_for_disk(a, run_dir)
    policy = pick()  # the newest export again: the round above can take a while
    out = run_dir / "rollouts" / f"c{a.collector_id}_{k:04d}_{policy.name}"
    server = Server(a, policy, run_dir / "logs" / f"serve_c{k:03d}.log")
    try:
        cmd = taskset(a.rollout_cpus) + [str(SIM_PY), "scripts/rollout.py", "--server", f"ws://localhost:{a.port}", "--out", str(out),
               "--episodes-per-task", str(a.episodes_per_task), "--seed-start", str(collect_seed_start(a, k)),
               "--workers", str(a.workers), "--gpus", a.render_gpus, "--record", *a.rollout_args.split()]
        cmd += ["--param-search", str(run_dir / "param_search.json")]  # Thompson-sampled inference / client params
        sw = policy / "score_weights.json"
        if sw.exists():  # monitoring: each episode's linearised score under the current weights
            cmd += ["--score-weights", str(sw)]
        with open(run_dir / "logs" / f"rollout_c{k:03d}.log", "w") as f:
            r = subprocess.run(cmd, cwd=SOL, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"rollout {k} failed")
        (out / "READY").touch()  # complete: bucket_daemon (collector role) pushes it
        summary = json.loads((out / "summary.json").read_text())
        entry = {"collect": k, "policy": str(policy), "run": str(out), "score": summary["score"]["overall"],
                 "success": {f: m["success"] for f, m in summary["per_family"].items()}, "time": time.time()}
        with state.lock:
            state.d["collected"].append(entry)
        state.save()
        log.info("collect %d (%s): RDS %.3f, success %s", k, policy.name, entry["score"].get("score", 0.0),
                 {f: round(v, 2) for f, v in entry["success"].items()})
    finally:
        server.stop()
    return out


def finalize_partial(a, state: State, run_dir: Path) -> None:
    """Keep a collect whose rollout was killed (low memory, reboot, restart): its finished episodes are complete on
    disk, so it gets summary.json + param_search_best.json (rollout.py --finalize), its state entry and READY,
    and the next iteration runs its recovery cascade like any collect's. Under --min-partial episodes: deleted."""
    known = {c["run"] for c in state.d["collected"]}
    for out in sorted((run_dir / "rollouts").glob(f"c{a.collector_id}_*"), key=lambda p: p.name):
        if str(out) in known or not (out / "episodes.jsonl").exists():
            continue
        n = sum(1 for _ in open(out / "episodes.jsonl"))
        if n < a.min_partial:
            log.warning("collect %s was cut short at %d episodes (< --min-partial %d): deleted", out.name, n,
                        a.min_partial)
            shutil.rmtree(out)
            continue
        if not (out / "summary.json").exists():
            r = subprocess.run([str(SIM_PY), "scripts/rollout.py", "--finalize", str(out), "--param-search",
                                str(run_dir / "param_search.json")], cwd=SOL, capture_output=True, text=True)
            if r.returncode != 0:
                log.error("finalising %s failed: %s", out.name, r.stdout[-2000:] + r.stderr[-2000:])
                continue
        summary = json.loads((out / "summary.json").read_text())
        k = int(out.name.split("_")[1])
        policy = run_dir / "policies" / out.name.split("_", 2)[2]
        entry = {"collect": k, "policy": str(policy), "run": str(out), "score": summary["score"]["overall"],
                 "success": {f: m["success"] for f, m in summary["per_family"].items()}, "time": time.time(),
                 "partial": n}
        with state.lock:
            state.d["collected"].append(entry)
        state.save()
        (out / "READY").touch()
        log.info("collect %d (%s): cut short, kept %d episodes: RDS %.3f, success %s", k, policy.name, n,
                 entry["score"].get("score", 0.0), {f: round(v, 2) for f, v in entry["success"].items()})


def clean_stale_recovery(a, run_dir: Path) -> None:
    """Recovery runs of this collector without READY never get pushed: cut short by a restart (the next run of that
    collect may use a newer policy, i.e. another dir name) or failed. Their files only fill the disk."""
    for d in sorted((run_dir / "recovery").glob(f"r{a.collector_id}_*")):
        if d.is_dir() and not (d / "READY").exists():
            shutil.rmtree(d, ignore_errors=True)
            log.info("removed recovery run %s (no READY: cut short or failed)", d.name)


def pending_recovery(state: State) -> dict | None:
    """The newest collect, if its recovery cascade has not run yet (and its rollouts are still on disk)."""
    if not state.d["collected"]:
        return None
    last = state.d["collected"][-1]
    if last["run"] in state.d.get("recovered", []):
        return None
    if not (Path(last["run"]) / "stores").is_dir():
        log.info("collect %d: rollouts no longer on disk, no recovery", last["collect"])
        return None
    return last


def recover(a, state: State, run_dir: Path, collect_dir: Path, k: int, policy: Path, env: dict) -> None:
    """Hard-case recovery cascade over collect k's failures (rollout.py --recover) with the served policy -> runs/
    <name>/recovery/r<id>_<k>_<policy>/ (recovery.json names the collect, which pairs them in training) + READY
    (bucket_daemon pushes it on its own). Never stops the collector: a crash or a run over --recovery-timeout-h is
    logged and the cycle goes on."""
    rdir = run_dir / "recovery" / f"r{a.collector_id}_{k:04d}_{policy.name}"
    if rdir.exists() and not (rdir / "READY").exists():  # a run cut short by a restart
        shutil.rmtree(rdir)
    cmd = taskset(a.rollout_cpus) + [str(SIM_PY), "scripts/rollout.py", "--server", f"ws://localhost:{a.port}",
                                     "--out", str(rdir), "--recover", str(collect_dir),
                                     "--recover-seed", str(k), "--recover-policy", policy.name,
                                     "--workers", str(a.recovery_workers), "--gpus", a.render_gpus, "--record",
                                     "--store-success-only", "--env-cache", "2"]
    if a.recovery_max:
        cmd += ["--recover-max", str(a.recovery_max)]
    cmd += ["--recover-track", str(a.recovery_track)]
    t0 = time.time()
    try:
        with open(run_dir / "logs" / f"recovery_c{k:03d}.log", "w") as f:
            r = subprocess.run(cmd, cwd=SOL, env=env, stdout=f, stderr=subprocess.STDOUT,
                               timeout=a.recovery_timeout_h * 3600 if a.recovery_timeout_h > 0 else None)
        # exit 2: cut short (a lost worker); every finished episode is valid and summary.json is written, so the
        # partial run is kept like a complete one
        if r.returncode not in (0, 2) or not (rdir / "summary.json").exists():
            log.error("recovery for collect %d failed (see logs/recovery_c%03d.log)", k, k)
        else:
            if r.returncode == 2:
                log.error("recovery for collect %d cut short (see logs/recovery_c%03d.log): keeping its finished "
                          "episodes", k, k)
            rs = json.loads((rdir / "summary.json").read_text()).get("recovery", {})
            if (rdir / "left.jsonl").exists():  # the unrecovered targets' seeds, kept after the run leaves the disk
                with open(run_dir / "recovery_left.jsonl", "a") as f:
                    for line in (rdir / "left.jsonl").read_text().splitlines():
                        f.write(json.dumps({"collect": k, "recovery": rdir.name, **json.loads(line)}) + "\n")
            (rdir / "READY").touch()
            log.info("recovery %d (%s): %d/%d failed / short episodes recovered (%d/%d targets incl. track chains) in "
                     "%.0fs %s, left %s", k, policy.name, rs.get("episodes_recovered", 0),
                     rs.get("episodes", 0), rs.get("recovered", 0), rs.get("failures", 0),
                     time.time() - t0,
                     {s: f"{v['successes']}/{v['tries']}" for s, v in rs.get("per_stage", {}).items()}, rs.get("left"))
    except subprocess.TimeoutExpired:
        log.error("recovery for collect %d killed after %.1f h (--recovery-timeout-h; see logs/recovery_c%03d.log)",
                  k, a.recovery_timeout_h, k)
    except Exception:  # noqa: BLE001
        log.exception("recovery for collect %d crashed", k)
    with state.lock:  # tried once, whatever the outcome
        state.d.setdefault("recovered", []).append(str(collect_dir))
    state.save()


def load_config(p: argparse.ArgumentParser, argv: list[str] | None = None) -> None:
    """--config <yaml>: its keys (flag names, with dashes or underscores) become the parser's defaults, so the command
    line still overrides them. Unknown keys are an error."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    path = pre.parse_known_args(argv)[0].config
    if not path:
        return
    import yaml

    cfg = {k.replace("-", "_"): v for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items()}
    actions = {a.dest: a for a in p._actions}
    unknown = sorted(set(cfg) - set(actions))
    if unknown:
        raise SystemExit(f"{path}: unknown keys {unknown}")
    for k in cfg:
        actions[k].required = False  # given by the config; the command line may still override it
    p.set_defaults(**cfg)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="", help="YAML file of flag defaults (configs/*.yaml)")
    p.add_argument("--name", required=True)
    p.add_argument("--base-config", default="pi05_roboeval")
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--role", choices=["trainer", "collector"], required=True,
                   help="trainer / collector machines, synced by bucket_daemon.py")
    # trainer
    p.add_argument("--rounds", type=int, default=10, help="trainer: at most this many train.py launches (each resumes)")
    p.add_argument("--steps-per-round", type=int, default=1000,
                   help="trainer: training steps per LeHome share-decay round (flywheel steps_per_decay)")
    p.add_argument("--total-steps", type=int, default=100_000, help="trainer: training steps (= the LR decay horizon)")
    p.add_argument("--stop-step", type=int, default=0,
                   help="trainer: stop at this step instead of --total-steps (the LR schedule keeps its horizon)")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--peak-lr", type=float, default=0.0, help="trainer: the cosine schedule's peak LR "
                   "(0 = the config's; with a larger --batch-size scale it by sqrt(new / old batch), the Adam rule)")
    p.add_argument("--decay-lr", type=float, default=0.0, help="trainer: the schedule's final LR (0 = the config's)")
    p.add_argument("--cooldown-start", type=int, default=0,
                   help="trainer: from this step a cosine from the schedule's LR there down to --cooldown-lr over "
                        "--cooldown-steps, then constant (0 = off; training/config.CosineDecayWithCooldown)")
    p.add_argument("--cooldown-steps", type=int, default=2500)
    p.add_argument("--cooldown-lr", type=float, default=3e-6)
    p.add_argument("--num-workers", type=int, default=0, help="data loader workers (0 = config default)")
    p.add_argument("--save-interval", type=int, default=250, help="trainer: checkpoint + policy export every N steps")
    p.add_argument("--train-mem", type=float, default=0.9)
    p.add_argument("--demo-weight", type=float, default=0.5,
                   help="trainer: the demos' sample_share (every dataset is sampled in proportion to its frames x its "
                        "sample_share; a fresh collect's is 1; 0 = demos not trained on)")
    p.add_argument("--demo-advantage", type=float, default=-1.0,
                   help="trainer: advantage of every demo frame (-1: the demos stay a low-advantage behaviour that "
                        "classifier-free guidance steers away from)")
    p.add_argument("--rollout-decay", type=float, default=0.8,
                   help="trainer: sample_share of a collect / recovery run / scripted generation: x this per "
                        "--steps-per-round training steps since it joined")
    p.add_argument("--pool-window", type=int, default=6, help="trainer: train on the newest N collects")
    p.add_argument("--recovery-window", type=int, default=6,
                   help="trainer: the newest N hard-case recovery runs (runs/<name>/recovery/*/ with DONE), one "
                        "advantage-sampled source each; 0 = off")
    p.add_argument("--scripted-window", type=int, default=8,
                   help="trainer: scripted generations kept in training, newest first; each decays like a collect")
    # collector
    p.add_argument("--collector-id", type=int, default=0, help="distinct per collector machine (seeds, names)")
    p.add_argument("--max-collects", type=int, default=0, help="collector: stop after N collects (0 = forever)")
    p.add_argument("--max-unpushed", type=int, default=2,
                   help="collector: pause while this many finished collects wait for bucket_daemon (0 = never)")
    p.add_argument("--min-free-gb", type=int, default=45,
                   help="collector: free disk (GiB) a collect waits for (0 = no check)")
    p.add_argument("--recovery-timeout-h", type=float, default=3.0,
                   help="collector: a recovery cascade running longer is killed and the cycle goes on (0 = none)")
    p.add_argument("--min-partial", type=int, default=200,
                   help="collector: a collect cut short with at least this many episodes is kept (else deleted)")
    p.add_argument("--recovery-workers", type=int, default=16)
    p.add_argument("--between-cmd", default="", help="collector: shell command run before each collect, after the "
                   "previous collect's recovery (e.g. scripts/scripted_round.sh; {name} = the run name), once per "
                   "collect number, with no policy server running (one GPU job at a time)")
    p.add_argument("--recovery-max", type=int, default=0, help="collector: failures per collect (random; 0 = all)")
    p.add_argument("--recovery-track", type=int, default=8,
                   help="collector: rollout.py --recover-track (track stage: onset-search pool size; 0 = off)")
    p.add_argument("--episodes-per-task", type=int, default=30)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--render-gpus", default="0",
                   help="EGL devices for the rollout workers, round-robin. Renders on the policy GPU steal its "
                        "inference time; with a second GPU put most rendering there (e.g. 0,1,1,1)")
    p.add_argument("--server-cpus", default="", help="taskset CPU list for the policy server (e.g. 0,1,12,13)")
    p.add_argument("--rollout-cpus", default="", help="taskset CPU list for rollout workers (e.g. 2-11,14-23)")
    p.add_argument("--seed-base", type=int, default=2_000_000)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-batch", type=int, default=48)
    p.add_argument("--serve-mem", type=float, default=0.3)
    p.add_argument("--serve-args", default="")
    p.add_argument("--rollout-args", default="")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = make_parser()
    load_config(p, argv)
    a = p.parse_args(argv)
    a.between_cmd = a.between_cmd.replace("{name}", a.name)  # a config can't know the run name
    return a


def main(argv: list[str] | None = None):
    a = parse_args(argv)
    if a.role == "trainer" and a.total_steps <= 0:
        raise SystemExit("--total-steps must be positive")

    run_dir = Path(a.runs_dir) / a.name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=log_handlers(run_dir / "logs" / "rl_loop.log"))
    (run_dir / "args.json").write_text(json.dumps(vars(a), indent=1))
    state = State(run_dir / "state.json")
    sys.path.insert(0, str(SOL / "src"))

    if a.role == "trainer":  # train.py on demos + bucket-pulled data; exports pushed by train.py / the daemon
        for rnd in range(len(state.d["exports"]), a.rounds):
            if train_round(a, state, rnd) is None:
                break
        return
    # collector: roll out the newest pulled policy forever
    finalize_partial(a, state, run_dir)
    clean_stale_recovery(a, run_dir)
    k = len(state.d["collected"])
    pdir = run_dir / "policies"
    while not a.max_collects or k < a.max_collects:
        if newest_export(pdir) is None:
            log.info("no policy yet in %s (bucket_daemon pulls it); waiting", pdir)
            time.sleep(60)
            continue
        unpushed = [r for r in (run_dir / "rollouts").glob("*") if (r / "READY").exists()
                    and not (r / "PUSHED").exists()]
        if a.max_unpushed and len(unpushed) >= a.max_unpushed:  # uploads lag: don't fill the disk
            log.info("%d collects waiting for upload; pausing", len(unpushed))
            time.sleep(60)
            continue
        collect(a, state, lambda: newest_export(pdir), k)  # each job picks the newest pulled export
        k += 1


if __name__ == "__main__":
    main()
