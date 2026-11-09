# Patches to third-party submodules

`openpi/` (Physical-Intelligence/openpi @ c23745b) and `RoboEval/` (Robo-Eval/RoboEval @ corl2026-challenge,
c8b7959) are pinned git submodules and are never edited in place. Any change goes here as
`patches/<submodule>/NN-description.patch` (`git -C <submodule> diff > patches/<submodule>/NN-x.patch`)
and is applied idempotently by `setup.sh`.

- `openpi/01-latest-jax-stack.patch`: openpi on the current JAX stack (jax 0.12, jaxtyping 0.3, flax 0.12):
  the jaxtyping dataclass check (a wrapped `__init__` in jaxtyping 0.3) is skipped while JAX / flax re-create
  pytree nodes during tracing, and the mesh asks for Auto axes (newer JAX defaults to Explicit, which breaks
  vmaps over mixed-sharded inputs).
- `openpi/02-orbax-step-metadata.patch`: `restore_params` with orbax >= 0.11, whose checkpoint metadata is a
  StepMetadata wrapping the tree metadata (policy serving and `distributed/export.py` load params through it).

RoboEval is used unmodified. Keep it that way unless strictly necessary — the official evaluator
fingerprints its source and a modified tree fails `roboeval.challenge.validate`.
