"""Collector / trainer orchestration: bucket reads that fail, scripted retirement on the trainer, recovery stores on the
collector, the between-collects round once per collect, and scripts/scripted_round.sh's generations."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from roboeval_solution.distributed import bucket

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── bucket reads ─────────────────────────────────────────────────────────────────────────────────────────────────────
def test_get_text_tells_absent_from_failed(monkeypatch):
    def run(rc, out="", err=""):
        return lambda *a, **k: subprocess.CompletedProcess([], rc, out, err)

    monkeypatch.setattr(bucket, "_run", run(0, "step_000500"))
    assert bucket.get_text("policies/x/LATEST") == "step_000500"
    monkeypatch.setattr(bucket, "_run", run(1, err="Error: 1 file(s) not found in bucket 'a/b': scripted/x/RETIRED.txt"))
    assert bucket.get_text("scripted/x/RETIRED.txt") is None
    monkeypatch.setattr(bucket, "_run", run(1, err="HTTP 429 Too Many Requests"))
    with pytest.raises(bucket.BucketError):  # never "absent": an unread RETIRED.txt would un-retire every chunk
        bucket.get_text("scripted/x/RETIRED.txt")
    monkeypatch.setattr(bucket, "_run", run(0, ""))
    assert bucket.list_dir("recovery/new_run") == []
    monkeypatch.setattr(bucket, "_run", run(1, err="connection reset"))
    with pytest.raises(bucket.BucketError):
        bucket.list_dir("rollouts/x")


class FakeBucket:
    """files: {remote path: text}; a remote "dir" is any prefix of a file path. pull_dir makes stores/."""

    def __init__(self, files, fail=()):
        self.files, self.fail, self.calls = dict(files), set(fail), []

    def list_dir(self, remote):
        self.calls.append(("list", remote))
        pre = remote.strip("/") + "/"
        return sorted({p[len(pre):].split("/")[0] for p in self.files if p.startswith(pre)})

    def get_text(self, remote):
        self.calls.append(("get", remote))
        if remote in self.fail:
            raise bucket.BucketError(remote)
        return self.files.get(remote)

    def pull_dir(self, remote, local):
        self.calls.append(("pull", remote))
        (Path(local) / "stores").mkdir(parents=True, exist_ok=True)

    def push_dir(self, local, remote, **kw):
        self.calls.append(("push", remote))

    def put_text(self, remote, text):
        self.calls.append(("put", remote))
        self.files[remote] = text


def _patch(monkeypatch, mod, fake):
    for f in ("list_dir", "get_text", "pull_dir", "push_dir", "put_text"):
        monkeypatch.setattr(mod.bucket, f, getattr(fake, f))


def _chunk(sdir, name, marker):
    (sdir / name / "stores").mkdir(parents=True)
    (sdir / name / marker).touch()


# ── trainer: scripted retirement ─────────────────────────────────────────────────────────────────────────────────────
def test_trainer_retires_by_local_dirs_and_never_repulls(monkeypatch, tmp_path):
    bd = _script("bucket_daemon")
    sdir = tmp_path / "scripted"
    _chunk(sdir, "g0001_1_a", "DONE")      # active, now named in RETIRED.txt
    _chunk(sdir, "g0001_2_a", "RETIRED")   # retired earlier (stores still in their grace hour)
    _chunk(sdir, "g0000_5_b", "DONE")      # retired, but a bucket clean-up already removed it from the listing
    files = {f"scripted/x/{n}/DONE": "1" for n in ("g0001_1_a", "g0001_2_a", "g0002_1_a")}
    files["scripted/x/RETIRED.txt"] = "g0001_1_a\ng0000_5_b\n"
    fake = FakeBucket(files)
    _patch(monkeypatch, bd, fake)
    bd.sync_scripted("x", tmp_path)
    assert (sdir / "g0001_1_a" / "RETIRED").exists() and not (sdir / "g0001_1_a" / "DONE").exists()
    assert (sdir / "g0000_5_b" / "RETIRED").exists()
    # g0001_2_a is no longer in RETIRED.txt (e.g. an overwritten list), but the local marker keeps it out
    assert ("pull", "scripted/x/g0001_2_a") not in fake.calls and not (sdir / "g0001_2_a" / "DONE").exists()
    assert ("pull", "scripted/x/g0002_1_a") in fake.calls and (sdir / "g0002_1_a" / "DONE").exists()


def test_trainer_pulls_nothing_scripted_when_retired_list_unreadable(monkeypatch, tmp_path):
    bd = _script("bucket_daemon")
    sdir = tmp_path / "scripted"
    _chunk(sdir, "g0001_1_a", "RETIRED")
    old = time.time() - bd.RETIRE_GRACE_S - 10
    os.utime(sdir / "g0001_1_a" / "RETIRED", (old, old))
    fake = FakeBucket({"scripted/x/g0001_1_a/DONE": "1", "scripted/x/g0002_1_a/DONE": "1"},
                      fail={"scripted/x/RETIRED.txt"})
    _patch(monkeypatch, bd, fake)
    bd.trainer_tick("x", tmp_path)  # never raises: each part of the tick fails on its own
    assert not [c for c in fake.calls if c[0] == "pull"]
    assert not (sdir / "g0001_1_a" / "stores").exists()  # local deletion still runs


def test_trainer_export_push_keeps_latest_moving_forward(monkeypatch, tmp_path):
    bd = _script("bucket_daemon")
    for d in ("params", "assets"):
        (tmp_path / "policies" / "step_000100" / d).mkdir(parents=True)
    fake = FakeBucket({"policies/x/LATEST": "step_000200"})
    _patch(monkeypatch, bd, fake)
    bd.push_exports("x", tmp_path)
    assert ("push", "policies/x/step_000100") in fake.calls and fake.files["policies/x/LATEST"] == "step_000200"
    assert (tmp_path / "policies" / ".pushed" / "step_000100").exists()


# ── collector: recovery stores ───────────────────────────────────────────────────────────────────────────────────────
def test_collector_drops_pushed_recovery_stores(monkeypatch, tmp_path):
    bd = _script("bucket_daemon")
    rdir = tmp_path / "recovery"
    for i, name in enumerate(("r0_0018_step_013000", "r0_0038_step_033500", "r0_0039_step_035000")):
        _chunk(rdir, name, "PUSHED")
        os.utime(rdir / name / "PUSHED", (1000 + i, 1000 + i))
    fake = FakeBucket({f"recovery/x/{n}/DONE": "1" for n in ("r0_0038_step_033500", "r0_0039_step_035000")})
    _patch(monkeypatch, bd, fake)
    bd.drop_recovery_stores("x", tmp_path, keep=1)
    assert not (rdir / "r0_0038_step_033500" / "stores").exists()   # in the bucket: local stores go
    assert (rdir / "r0_0039_step_035000" / "stores").exists()       # the newest stays
    # gone from the bucket (a clean-up): the local copy is the only one left
    assert (rdir / "r0_0018_step_013000" / "stores").exists() and (rdir / "r0_0018_step_013000" / "LOCAL_ONLY").exists()
    n = len(fake.calls)
    bd.drop_recovery_stores("x", tmp_path, keep=1)
    assert len(fake.calls) == n  # checked once


# ── collector: rl_loop ───────────────────────────────────────────────────────────────────────────────────────────────
def test_between_round_runs_once_per_collect(tmp_path):
    rl = _script("rl_loop")
    (tmp_path / "logs").mkdir()
    state = rl.State(tmp_path / "state.json")
    a = argparse.Namespace(between_cmd=f"echo round >> {tmp_path}/ran; echo 'round summary: x'; exit 1")
    rl.between_round(a, state, tmp_path, 7, dict(os.environ))
    rl.between_round(a, state, tmp_path, 7, dict(os.environ))  # a restart before collect 7 finished
    assert (tmp_path / "ran").read_text().count("round") == 1
    assert json.loads((tmp_path / "state.json").read_text())["between_done"] == 7
    rl.between_round(a, state, tmp_path, 8, dict(os.environ))
    assert (tmp_path / "ran").read_text().count("round") == 2


def test_stale_recovery_runs_are_removed(tmp_path):
    rl = _script("rl_loop")
    rec = tmp_path / "recovery"
    for name, ready in (("r0_0037_step_032500", True), ("r0_0038_step_033000", False), ("r1_0038_step_033000", False),
                        ("logs", False)):
        (rec / name).mkdir(parents=True)
        if ready:
            (rec / name / "READY").touch()
    rl.clean_stale_recovery(argparse.Namespace(collector_id=0), tmp_path)
    assert sorted(p.name for p in rec.iterdir()) == ["logs", "r0_0037_step_032500", "r1_0038_step_033000"]


# ── scripts/scripted_round.sh ────────────────────────────────────────────────────────────────────────────────────────
FAKE_SIM = """#!/bin/bash
# fake rollout.py --scripted: <out>/episodes.jsonl with N full-reward episodes; fails for a policy named in FAIL
while [ $# -gt 0 ]; do
  case $1 in --out) out=$2; shift ;; --episodes-per-task) n=$2; shift ;; --scripted) pol=$2; shift ;;
    --seed-start) seed=$2; shift ;; esac
  shift
done
echo "$pol $seed $n" >> "$(dirname "$0")/../../calls"
grep -qw "$pol" "$(dirname "$0")/../../FAIL" 2>/dev/null && exit 1
mkdir -p "$out"
for ((i = 0; i < n; i++)); do echo '{"success": true, "metrics": {"tp": 1.0}}'; done > "$out/episodes.jsonl"
"""
FAKE_PY = """#!/bin/bash
# fake training venv python: scripts/scripted_data.py retire ... (exit 1 when RETIRE_FAILS exists)
echo "$*" >> "$(dirname "$0")/../../retire_calls"
[ -e "$(dirname "$0")/../../RETIRE_FAILS" ] && exit 1
exit 0
"""


@pytest.fixture
def round_env(tmp_path):
    sol = tmp_path / "sol"
    (sol / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPTS / "scripted_round.sh", sol / "scripts")
    for venv, body in ((".venv-sim", FAKE_SIM), (".venv", FAKE_PY)):
        (sol / venv / "bin").mkdir(parents=True)
        (sol / venv / "bin" / "python").write_text(body)
        (sol / venv / "bin" / "python").chmod(0o755)
    run = tmp_path / "runs" / "R"
    run.mkdir(parents=True)

    def plan(**kw):
        base = dict(first=3, per=2, part=1, window=2, min_free_gb=0, max_unpushed=1000, max_wait_min=0,
                    policies="packbox stack", inject="off")
        base.update(kw)
        (run / "scripted_plan.txt").write_text("".join(f"{k} {v}\n" for k, v in base.items()))

    def go():
        r = subprocess.run(["bash", str(sol / "scripts" / "scripted_round.sh"), "R", "1", "0", "0"],
                           capture_output=True, text=True)
        return r.returncode, r.stdout.strip().splitlines()[-1]

    return run / "scripted", sol, plan, go


def _chunks(sdir, g):
    return sorted(p.name for p in sdir.glob(f"g{g:04d}_*") if (p / "READY").exists())


def test_scripted_round_generations(round_env):
    sdir, sol, plan, go = round_env
    plan()
    rc, last = go()
    assert rc == 0 and "generation 1 complete" in last
    assert _chunks(sdir, 1) == [f"g0001_{p}_{pol}" for p in (1, 2, 3) for pol in ("packbox", "stack")]
    assert (sdir / "g0001.plan").read_text() == "policies packbox stack\nepisodes 3\npart 1\n"
    # adding a policy (and changing the size) applies from the next generation: generation 1 stays as it was
    plan(policies="packbox stack pickbook nosuchpolicy", per=1)
    rc, last = go()
    assert rc == 0 and "generation 2 complete" in last
    assert _chunks(sdir, 1) == [f"g0001_{p}_{pol}" for p in (1, 2, 3) for pol in ("packbox", "stack")]
    assert _chunks(sdir, 2) == ["g0002_1_packbox", "g0002_1_pickbook", "g0002_1_stack"]
    assert (sdir / "g0002.plan").read_text().startswith("policies packbox stack pickbook\n")
    # generation 3 retires generation 1 (window 2), by generation number
    rc, last = go()
    assert rc == 0 and (sdir / "g0001.retired").exists()
    assert (sol / "retire_calls").read_text().split("\n")[0].endswith("retire --run R --generation 1")


def test_scripted_round_inject_and_unused_keys(round_env):
    """"inject <policy> N" (every generation): N injected-hard-case jobs as chunks g<G>_9<k>_<policy> of at most PART,
    seeds 120M + task * 1M + G * 20000 + chunk * 5000, part of the generation; unknown plan keys are logged and
    ignored."""
    sdir, sol, plan, go = round_env
    plan(first=1, per=1, part=2, extra="stack 3", inject="stack 3")
    r = subprocess.run(["bash", str(sol / "scripts" / "scripted_round.sh"), "R", "1", "0", "0"], capture_output=True,
                       text=True)
    assert r.returncode == 0 and "generation 1 complete" in r.stdout.strip().splitlines()[-1]
    assert "plan keys not used (ignored): extra" in r.stdout
    assert _chunks(sdir, 1) == ["g0001_1_packbox", "g0001_1_stack", "g0001_91_stack", "g0001_92_stack"]
    assert (sdir / "g0001.plan").read_text().endswith("inject stack 3\n")
    calls = (sol / "calls").read_text().split("\n")
    assert "stack 122020000 2" in calls and "stack 122025000 1" in calls and "packbox 70210000 1" in calls
    # an interrupted generation resumes its missing inject chunk only
    (sdir / "g0001_92_stack" / "READY").unlink()
    rc, last = go()
    assert rc == 0 and "generation 1 complete; 1 chunks made" in last


def test_scripted_round_failures_exit_nonzero(round_env):
    sdir, sol, plan, go = round_env
    plan(first=1)
    (sol / "FAIL").write_text("stack")
    rc, last = go()
    assert rc == 1 and "generation 1 incomplete" in last and "1 failed" in last
    (sol / "FAIL").unlink()
    rc, last = go()  # the next round finishes generation 1: only the failed chunk is made
    assert rc == 0 and "generation 1 complete; 1 chunks made" in last
    for _ in range(2):
        assert go()[0] == 0
    (sol / "RETIRE_FAILS").touch()
    rc, last = go()  # generation 4 retires generation 2
    assert rc == 1 and "retiring generation 2 failed" in last and not (sdir / "g0002.retired").exists()
    # no room (an upload backlog) after max_wait_min: the round gives up and the collect goes ahead
    (sol / "RETIRE_FAILS").unlink()
    plan(first=1, max_unpushed=1)
    rc, last = go()
    assert rc == 1 and "gave up waiting" in last
