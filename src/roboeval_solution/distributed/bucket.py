"""Hugging Face Storage Bucket as the trainer <-> rollout-collector bus.

Layout of hf://buckets/<bucket>/ (the bucket: RBE_BUCKET=<hf-user>/<bucket>, required; create it once with
`hf buckets create <hf-user>/<bucket> --private`):

    policies/<run>/step_<S>/          exported params (bf16) + assets             (trainer writes)
    policies/<run>/LATEST             newest exported step                        (collectors poll)
    checkpoints/<config>/<run>/<S>/   full training checkpoints, every N steps    (train.py)
    rollouts/<run>/<collect>/         collect run dir + DONE marker               (collectors write)
    recovery/<run>/<run id>/          hard-case recovery run + DONE               (collectors write)
    scripted/<run>/<chunk>/           scripted generation chunk g<GGGG>_<part>_<policy> + DONE (collectors write)
    scripted/<run>/RETIRED.txt        chunks out of training (scripts/scripted_data.py retire)

All operations shell out to a pinned `hf` CLI through `uvx`, so neither venv needs a
bucket-capable huggingface_hub (lerobot pins an older one). Standard library only.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

BUCKET = os.environ.get("RBE_BUCKET", "")
HF_CLI = os.environ.get("RBE_HF_CLI", "uvx --from huggingface_hub==2.3.0 hf").split()


class BucketError(RuntimeError):
    """A bucket read that failed for another reason than the file not existing (network, rate limit, auth)."""


def uri(path: str) -> str:
    if not BUCKET:
        raise BucketError("no bucket: set RBE_BUCKET=<hf-user>/<bucket> (a private Hugging Face Storage Bucket)")
    return f"hf://buckets/{BUCKET}/{path.strip('/')}"


LOG = os.environ.get("RBE_BUCKET_LOG", os.path.join(tempfile.gettempdir(), "rbe_bucket_cli.log"))
# Per-transfer timeout: a stalled transfer is killed and retried by the caller.
TIMEOUT_S = float(os.environ.get("RBE_BUCKET_TIMEOUT", "3600"))


def _run(args: list[str], capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(HF_CLI + args, check=check, text=True, capture_output=True, stdin=subprocess.DEVNULL,
                              timeout=300)
    with open(LOG, "a") as log:  # never inherit the caller's pipes
        return subprocess.run(HF_CLI + args, check=check, text=True, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                              timeout=TIMEOUT_S)


def push_dir(local: str | Path, remote: str, exclude: list[str] | None = None) -> None:
    """rsync-like upload of a local directory to bucket prefix `remote`."""
    args = ["buckets", "sync", str(local), uri(remote), "--format", "quiet"]
    for e in exclude or []:
        args += ["--exclude", e]
    _run(args)


def pull_dir(remote: str, local: str | Path) -> None:
    """rsync-like download of bucket prefix `remote` into a local directory."""
    Path(local).mkdir(parents=True, exist_ok=True)
    _run(["buckets", "sync", uri(remote), str(local), "--format", "quiet"])


def put_text(remote: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        _run(["cp", tmp, uri(remote), "--format", "quiet"])
    finally:
        os.unlink(tmp)


def get_text(remote: str) -> str | None:
    """The file's text, or None when the bucket has no such file. Any other failure raises BucketError: a caller must
    never read "unreadable" as "absent" (an unreadable RETIRED.txt taken as empty un-retires every chunk)."""
    r = _run(["cp", uri(remote), "-"], capture=True, check=False)
    if r.returncode == 0:
        return r.stdout
    err = (r.stderr or "") + (r.stdout or "")
    if "not found in bucket" in err:  # huggingface_hub EntryNotFoundError: "... file(s) not found in bucket ..."
        return None
    raise BucketError(f"reading {remote} failed (exit {r.returncode}): {err.strip()[-500:]}")


def list_dir(remote: str) -> list[str]:
    """Immediate children (files and folders) of a bucket prefix, as names ([] for a prefix with nothing under it).
    A failed listing raises BucketError (an empty list would read as "nothing there")."""
    r = _run(["buckets", "list", uri(remote), "--format", "json"], capture=True, check=False)
    if r.returncode != 0:
        raise BucketError(f"listing {remote} failed (exit {r.returncode}): {(r.stderr or '').strip()[-500:]}")
    if not r.stdout.strip():
        return []
    prefix = remote.strip("/") + "/"  # the CLI prints a JSON list of entries with their full bucket path
    names = [it["path"][len(prefix):].strip("/").split("/")[0] for it in json.loads(r.stdout)]
    return sorted(set(n for n in names if n))
