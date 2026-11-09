#!/usr/bin/env bash
# One-shot setup of this repo on a new machine.
#
#   bash setup.sh                 # submodules + patches + training env (.venv) + simulator env (.venv-sim)
#   bash setup.sh --no-sim        # training-only machine (no MuJoCo env)
#   bash setup.sh --data          # also download the public demos (~96 GB), convert them to rbe stores
#                                 # (5 cameras) and compute the norm-stats / FAST assets (scripts/prepare_assets.py)
#   bash setup.sh --data --keep-raw   # keep the raw LeRobot download after conversion
#   bash setup.sh --cuda12        # NVIDIA driver < 580 (no CUDA 13): JAX with its CUDA 12 plugin; then run the
#                                 # training env with UV_NO_SYNC=1 (`uv run` re-syncs to the lockfile and would
#                                 # reinstall the CUDA 13 plugin next to it) or .venv/bin/python
#
# Layout: the repo lives in <root>/<repo>; data/, outputs/, runs/, logs/ are created next to it (in <root>).
# Third-party code is pinned as git submodules and never edited in place: changes go to
# patches/<submodule>/*.patch and are applied here (idempotent).
set -euo pipefail

SIM=1; DATA=0; KEEP_RAW=0; CUDA12=0
for arg in "$@"; do
  case "$arg" in
    --no-sim) SIM=0 ;;
    --data) DATA=1 ;;
    --keep-raw) KEEP_RAW=1 ;;
    --cuda12) CUDA12=1 ;;
    *) echo "unknown option $arg"; exit 1 ;;
  esac
done

cd "$(dirname "$0")"
REPO=$PWD
ROOT=$(dirname "$REPO")
command -v uv >/dev/null || { echo "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

echo "== submodules"
git submodule update --init openpi RoboEval      # openpi @ c23745b, RoboEval @ corl2026-challenge
git -C RoboEval submodule update --init          # mujoco_menagerie (robot meshes)

echo "== patches"
for lib in openpi RoboEval; do
  for p in "$REPO"/patches/"$lib"/*.patch; do
    [ -e "$p" ] || continue
    if git -C "$lib" apply --reverse --check "$p" 2>/dev/null; then
      echo "already applied: ${p#"$REPO"/}"
    else
      git -C "$lib" apply "$p" && echo "applied: ${p#"$REPO"/}"
    fi
  done
done

mkdir -p "$ROOT"/data/stores "$ROOT"/outputs "$ROOT"/runs "$ROOT"/logs

echo "== training / inference env (.venv, python 3.12)"
UV_HTTP_TIMEOUT=300 uv sync
if [ "$CUDA12" = 1 ]; then  # NVIDIA driver < 580: the same JAX with its CUDA 12 plugin instead of CUDA 13
  JAXV=$(.venv/bin/python -c "import jax; print(jax.__version__)")
  VIRTUAL_ENV=.venv uv pip uninstall -q jax-cuda13-plugin jax-cuda13-pjrt
  VIRTUAL_ENV=.venv uv pip install -q "jax[cuda12]==$JAXV"
fi

if [ "$SIM" = 1 ]; then
  echo "== simulator env (.venv-sim, python 3.10: roboeval + mujoco 3.3.3)"
  [ -x .venv-sim/bin/python ] || uv venv .venv-sim --python 3.10 -q
  # = RoboEval's pyproject dependencies (+ what the rollout workers need), numpy pinned last. --no-config: the repo's
  # [tool.uv] overrides (numpy>=2 for the training env) must not apply here; RoboEval pins numpy 1.26.
  # uv >= 0.7 rejects the stepjam Gymnasium fork's duplicate extras (classic-control / classic_control): pin an older uv.
  UV_HTTP_TIMEOUT=300 VIRTUAL_ENV=.venv-sim uvx uv@0.6.17 pip install --no-config -q \
    "mujoco==3.3.3" "dm_control==1.0.31" "safetensors==0.4.3" imageio pyquaternion mujoco_utils wget pyyaml \
    "hydra-core==1.3.*" psutil scipy "opencv-python-headless<4.12" pyarrow pandas pillow \
    "gymnasium @ git+https://github.com/stepjam/Gymnasium.git@0.29.2" \
    "mojo @ git+https://github.com/helen9975/mojo.git" \
    "roboeval-metrics @ git+https://github.com/Robo-Eval/roboeval-metrics.git" \
    dearpygui pyopenxr click-prompt rich-click "pynput==1.7.7" pytest \
    -e openpi/packages/openpi-client
  VIRTUAL_ENV=.venv-sim uv pip install -q --no-deps -e ./RoboEval
  VIRTUAL_ENV=.venv-sim uv pip install --no-config -q "numpy==1.26.*"
  MUJOCO_GL=egl .venv-sim/bin/python -c "import roboeval, mujoco, numpy; print('sim env ok: mujoco', mujoco.__version__, 'numpy', numpy.__version__)"
fi

if [ "$DATA" = 1 ]; then
  RAW="$ROOT/data/roboeval_ee_delta_20hz_scaled"
  STORES="$ROOT/data/stores/demos"
  N_VARIATIONS=30  # 8 families x their variations; one store each
  converted() { local n=0 f; for f in "$STORES"/*/meta.json; do [[ -e "$f" && "$f" != *.partial/* ]] && n=$((n + 1)); done; echo "$n"; }
  # (re)download + convert until every variation has its store: convert_dataset.py skips finished stores, so an
  # interrupted conversion resumes where it stopped instead of being skipped for good
  N_DONE=$(converted)
  if [ "$N_DONE" -lt "$N_VARIATIONS" ]; then
    echo "== public demos -> $RAW ($N_DONE/$N_VARIATIONS stores converted)"
    uvx --from huggingface_hub==2.3.0 hf download helen9975/roboeval_ee_delta_20hz_scaled --repo-type dataset --local-dir "$RAW"
    echo "== convert to rbe stores (5 cameras) -> $STORES"
    .venv/bin/python scripts/convert_dataset.py --src "$RAW" --out "$STORES" --workers 8
    N_DONE=$(converted)
    [ "$N_DONE" -ge "$N_VARIATIONS" ] || { echo "only $N_DONE/$N_VARIATIONS demo stores converted"; exit 1; }
    [ "$KEEP_RAW" = 1 ] || rm -rf "$RAW"
  fi
  echo "== assets (norm stats, chunk correlation, FAST tokenizer) computed from the demos"
  .venv/bin/python scripts/prepare_assets.py --config-name pi05_roboeval
fi

cat <<EOF

Done. Useful environment:
  export PATH=$REPO/.venv/lib/python3.12/site-packages/nvidia/cu13/bin:\$PATH   # ptxas for Blackwell GPUs
  export MUJOCO_GL=egl                                                                  # headless rendering
Training:  .venv/bin/python scripts/train.py pi05_roboeval --exp_name <name>
Rollouts:  .venv-sim/bin/python scripts/rollout.py --server ws://localhost:8765 --out ../runs/<name>
Tests:     .venv/bin/python -m pytest -q tests/
           .venv-sim/bin/python -m pytest -q tests/test_score_reward.py tests/test_metric_recorder.py tests/test_tasks.py tests/test_rollout.py
EOF
if [ "$CUDA12" = 1 ]; then
  echo "--cuda12: run the training env with UV_NO_SYNC=1 (uv run ...) or .venv/bin/python: a syncing 'uv run'"
  echo "          reinstalls the locked CUDA 13 JAX plugin next to the CUDA 12 one."
fi
