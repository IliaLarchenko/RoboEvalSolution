"""Per-group LR multipliers (following our BEHAVIOR-1K solution).

Groups partition the trainable params by role (vision tower, VLM trunk, action expert, ...). The multipliers
(``LRMultipliers``, TrainConfig.lr_multipliers) scale each param's final update after AdamW. They are stateless: the
optimizer state pytree is unchanged, so checkpoints restore across multiplier changes.

Path matching is against "/"-joined pytree key paths of the trainable-param state (e.g.
``PaliGemma/llm/layers/attn/q_einsum_1/w/value``). Rules are ordered; first match wins. The vision rule must precede
the expert rule: the action expert is identified by Gemma's ``_1`` module suffix, which also appears in SigLIP layer
names (``LayerNorm_1``, ``Dense_1``).
"""

import dataclasses
import re

import jax
import optax


@dataclasses.dataclass(frozen=True)
class LRMultipliers:
    """Per-group LR multipliers (1.0 = the base schedule). New small params (query tokens, heads, embeddings) can
    use higher multipliers to move off their init at a low base LR."""

    vision: float = 1.0        # SigLIP tower (PaliGemma/img/...)
    llm: float = 1.0           # Gemma VLM trunk (PaliGemma/llm minus the _1 expert)
    expert: float = 1.0        # action expert: Gemma modules with the _1 suffix
    action_proj: float = 1.0   # action/time projections in and out of the expert
    kv_transform: float = 1.0  # cross-layer KV mixing coefficients
    embeddings: float = 1.0    # task / advantage / state embeddings
    fast: float = 1.0          # FAST auxiliary token embedding + projection
    value_heads: float = 1.0   # value / Q heads, their query tokens and time fusions
    pe: float = 1.0            # per-camera projections, per-dim state embedding, state value/continuous projections
    other: float = 1.0         # params no rule matches


_GROUP_RULES: tuple[tuple[str, re.Pattern], ...] = (
    ("pe", re.compile(r"^(?:camera_proj_\w+|state_dim_embedding|state_value_proj|state_continuous_proj|velocity_continuous_proj)(?:/|$)")),
    ("vision", re.compile(r"^PaliGemma/img/")),
    ("expert", re.compile(r"_1(?:/|$)")),  # must come after vision (SigLIP has Dense_1 etc.)
    ("llm", re.compile(r"^PaliGemma/llm/")),
    ("action_proj", re.compile(r"^(?:action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/")),
    ("kv_transform", re.compile(r"^kv_transform/")),
    ("embeddings", re.compile(r"^(?:task_input_embedding|task_adarms_embedding|advantage_embeddings"
                              r"|advantage_adarms_vec|state_embedding)(?:/|$)")),
    ("fast", re.compile(r"^(?:fast_token_embedding|fast_token_proj)/")),
    ("value_heads", re.compile(r"^(?:success_query_token|success_head|task_head|completion_head"
                               r"|stage_head|togo_head|wm_fast_\w+|wm_flow_\w+|aux_time_\w+)(?:/|$)")),
)

GROUPS: tuple[str, ...] = tuple(name for name, _ in _GROUP_RULES) + ("other",)


def _key_str(entry) -> str:
    for attr in ("key", "name", "idx"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


def path_string(key_path) -> str:
    """A jax key path -> "PaliGemma/llm/.../w/value" style string."""
    return "/".join(_key_str(k) for k in key_path)


def group_of(path: str) -> str:
    for name, pattern in _GROUP_RULES:
        if pattern.search(path):
            return name
    return "other"


def with_lr_multipliers(
    base_tx: optax.GradientTransformation, multipliers: LRMultipliers
) -> optax.GradientTransformation:
    """Wrap ``base_tx`` and scale each param's final update by its group multiplier. The state is the base state;
    groups at exactly 1.0 stay bit-identical to the unwrapped optimizer."""
    mults = dataclasses.asdict(multipliers)
    unknown = set(mults) - set(GROUPS)
    if unknown:
        raise ValueError(f"LRMultipliers fields {sorted(unknown)} have no matching group rule")
    for name, value in mults.items():
        if value <= 0:
            raise ValueError(f"lr_multipliers.{name} must be > 0, got {value}")

    def update(updates, state, params=None):
        updates, new_state = base_tx.update(updates, state, params)
        flat, treedef = jax.tree_util.tree_flatten_with_path(updates)
        scaled = [
            leaf if (m := mults[group_of(path_string(kp))]) == 1.0 else leaf * m
            for kp, leaf in flat
        ]
        return jax.tree_util.tree_unflatten(treedef, scaled), new_state

    return optax.GradientTransformation(base_tx.init, update)
