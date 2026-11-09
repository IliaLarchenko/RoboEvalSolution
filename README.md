# Off-Policy RL with Online Inference Tuning: my RoboEval CoRL 2026 solution

A π0.5-based vision-language-action policy (openpi, JAX) for the
[RoboEval Diagnostic Bimanual Challenge](https://robo-eval.github.io/challenge.html) (CoRL 2026): eight two-arm
Franka tasks, scored by the RoboEval Diagnostic Score (RDS). The policy is also its own critic: it has value heads and
a Q head that ranks sampled action chunks. It is trained by an off-policy RL loop on advantages of the exact challenge
score, and every rollout also tunes about 20 inference settings by Thompson sampling. On top of that come two kinds
of extra data: scripted full-reward demos, and recovery episodes that branch off the policy's own failures.

| | RDS |
|---|---|
| Official challenge seeds (submitted policy) | **0.987** |
| Held-out seeds (500 episodes per task) | **0.990** |

* Paper: [arXiv:2611.11464](https://arxiv.org/abs/2611.11464) · Project page: [ilialarchenko.com/projects/roboeval2026](https://ilialarchenko.com/projects/roboeval2026/) ·
  [Demo video](https://youtu.be/3N8kYqua924)
* Checkpoint: [huggingface.co/IliaLarchenko/RoboEvalSolution](https://huggingface.co/IliaLarchenko/RoboEvalSolution)

## How it works

* **Model** (`src/roboeval_solution/models/`, config `pi05_roboeval`): π0.5 on five 256x256 cameras and the joint
  state, predicting 30-step chunks of end-effector poses. Extra inputs: a task token and an advantage token (for
  classifier-free guidance). Extra heads: success, stages, completion, behavior-to-go, and Q heads that rank sampled
  chunks.
* **Advantages** (`scripts/compute_advantages.py`, `data/score_reward.py`): the challenge score turned into a per-step
  reward, a baseline from the collecting policy's own value predictions, GAE.
* **RL loop** (`scripts/rl_loop.py`, `training/flywheel.py`): one trainer and one or more collectors exchange
  policies and data through a Hugging Face Storage Bucket. The trainer picks up new data without stopping.
* **Collection** (`scripts/rollout.py`, `scripts/serve_policy.py`): a batched policy server, parallel MuJoCo workers,
  the official score for every episode, and Thompson sampling over the inference settings.
* **Recovery** (`sim/recovery.py`): failed episodes are replayed to a branch point and continued by the policy or by
  a scripted controller; only full-reward continuations are kept.
* **Scripted demos** (`sim/scripted_*.py`, `scripts/scripted_round.sh`): controllers that read the simulator state and
  reach the full reward in more than 99.5% of episodes, generated every collector cycle.

## Setup

Linux x86-64 with NVIDIA GPUs (training: CUDA 13 / driver >= 580, or `--cuda12`).

```bash
mkdir -p ~/roboeval && cd ~/roboeval
git clone --recurse-submodules https://github.com/IliaLarchenko/RoboEvalSolution.git && cd RoboEvalSolution
bash setup.sh --no-sim --data   # trainer: .venv, demos converted to the training format, norm stats
bash setup.sh                   # collector: .venv and .venv-sim (the simulator)
```

* `setup.sh` builds both environments with [uv](https://docs.astral.sh/uv/). The commands below call
  `.venv/bin/python` directly, because `uv run` would re-sync to the lockfile.
* `--data` downloads the public demos (`helen9975/roboeval_ee_delta_20hz_scaled`, ~96 GB). The π0.5 base weights
  download on first use.
* Create a private Hugging Face Storage Bucket and point every machine at it: `export RBE_BUCKET=<hf-user>/<bucket>`
  (log in once with `uvx --from huggingface_hub==2.3.0 hf auth login`). Training logs to wandb (`wandb login`).
* Data, checkpoints and runs live next to the repo (`../data`, `../outputs`, `../runs`).

## Training

**Trainer:**

```bash
export RBE_BUCKET=<hf-user>/<bucket>
.venv/bin/python scripts/bucket_daemon.py --run <run> --role trainer --keep-collects 8 --keep-recovery 8 &
.venv/bin/python scripts/rl_loop.py --config configs/trainer.yaml --name <run>
```

`configs/trainer.yaml` is the final run's setup (8x B300). Any setting can be overridden on the command line, e.g.
`--batch-size 96 --peak-lr 2.5e-5` on one H200. The trainer's `train.py` runs on the GPUs listed in `RBE_TRAIN_GPU`
(default `0`); the final run used `export RBE_TRAIN_GPU=0,1,2,3,4,5,6,7`. To train on the demos alone, without the
loop: `.venv/bin/python scripts/train.py pi05_roboeval --exp_name <name>` (add `--no-bucket-upload` to keep
everything local; a full checkpoint takes ~42 GB of disk).

**Collector:**

```bash
export RBE_BUCKET=<hf-user>/<bucket>
.venv/bin/python scripts/bucket_daemon.py --run <run> --role collector --keep-rollouts 1 &
.venv/bin/python scripts/rl_loop.py --config configs/collector.yaml --name <run>
```

`configs/collector.yaml` is set up for a box with one GPU and 24 CPU cores. It runs forever: recovery of the previous
collect, one batch of scripted demos, then 2,000 episodes on the newest policy. More collectors: the same command with
`--collector-id 1 --between-cmd ''` (only one machine makes the scripted demos). State and logs go to `runs/<run>/`.

## Evaluation

```bash
.venv/bin/hf download IliaLarchenko/RoboEvalSolution --local-dir ../checkpoint   # the submitted policy
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/serve_policy.py --config-name pi05_roboeval \
    --checkpoint-dir ../checkpoint --num-candidates 6 &
MUJOCO_GL=egl .venv-sim/bin/python scripts/rollout.py --server ws://localhost:8765 --out ../runs/eval_x \
    --episodes-per-task 50 --workers 20 --best configs/settings.json
```

`rollout.py` builds the environments like the official evaluator, scores every episode with the official code
(`summary.json`) and refuses the official seeds 1000-1049. `--best` runs every task at the final per-task settings.

## Tests

```bash
.venv/bin/python -m pytest -q tests                  # training venv (CPU)
MUJOCO_GL=egl .venv-sim/bin/python -m pytest -q tests/test_score_reward.py tests/test_metric_recorder.py \
    tests/test_tasks.py tests/test_rollout.py        # simulator venv: checks against the official scorer
```

## License

The code and the checkpoint are released under the [Apache License 2.0](LICENSE). The submodules keep their own
licenses (openpi: Apache 2.0; RoboEval: see its repository).

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence: pi0.5 and its base
weights) and [RoboEval](https://github.com/Robo-Eval/RoboEval) (the benchmark, simulator, demonstrations and official
evaluator). The training recipe builds on my [LeHome Challenge 2026 solution](https://github.com/IliaLarchenko/lehome_solution).

## Citation

```bibtex
@misc{larchenko2026roboeval,
  title  = {Off-Policy RL with Online Inference Tuning: My Solution to the CoRL 2026 RoboEval Challenge},
  author = {Larchenko, Ilia},
  year   = {2026},
  eprint = {2611.11464},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url    = {https://arxiv.org/abs/2611.11464}
}
```
