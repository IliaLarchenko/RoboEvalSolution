"""Norm stats (+ correlation Cholesky) and FAST tokenizer for a RoboEval training config.

    .venv/bin/python scripts/prepare_assets.py --config-name pi05_roboeval [--num-chunks 200000]

Statistics are computed over action chunks sampled with the *training* sampling
distribution, in the model action representation:
  state   : mean/std; q01/q99 := min/max widened by --state-margin (velocity dims: the --state-vel-quantile
            percentiles); the state is quantile-mapped to [-1, 1]
  actions : global mean/std/q01/q99, per-timestep std, and the Cholesky factor of the
            per-timestep-normalised chunk correlation matrix.
Padded timesteps (past the episode end) are excluded.
Writes <assets_base_dir>/<config>/<asset_id>/{norm_stats.json, fast_tokenizer/}.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from roboeval_solution.shared import normalize
from roboeval_solution.training import config as _config
from roboeval_solution.training.config import get_config


def correlation_cholesky(chunks: np.ndarray, max_samples: int = 400_000) -> np.ndarray:
    n, H, D = chunks.shape
    flat = chunks.reshape(n, H * D).astype(np.float64)
    if n > max_samples:
        flat = flat[np.random.default_rng(0).choice(n, max_samples, replace=False)]
    mean, std = flat.mean(0), flat.std(0)
    const = std < 1e-6
    z = np.zeros_like(flat)
    z[:, ~const] = (flat[:, ~const] - mean[~const]) / std[~const]
    cov = np.cov(z, rowvar=False)
    d = np.diag(cov).copy()
    cmask = d < 1e-10
    d[cmask] = 1.0
    cov = cov / np.sqrt(d[:, None] * d[None, :])
    cov[cmask, :] = 0.0
    cov[:, cmask] = 0.0
    np.fill_diagonal(cov, 1.0)
    eps = 1e-6
    min_eig = np.linalg.eigvalsh(cov + eps * np.eye(len(cov))).min()
    if min_eig <= 0:
        eps = max(1e-5, -min_eig + 1e-5)
    print(f"correlation: {len(flat)} chunks, min eig {min_eig:.2e}, eps {eps:.1e}")
    return np.linalg.cholesky(cov + eps * np.eye(len(cov)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--num-chunks", type=int, default=200_000)
    p.add_argument("--fast-chunks", type=int, default=100_000)
    p.add_argument("--fast-scale", type=float, default=10.0)
    p.add_argument("--state-margin", type=float, default=0.02, help="fraction of the range added on both sides")
    p.add_argument("--state-vel-quantile", type=float, default=0.005,
                   help="velocity state dims are scaled to this lower / upper percentile, not min/max")
    a = p.parse_args()

    cfg = get_config(a.config_name)
    factory = cfg.data
    if not isinstance(factory, _config.RoboEvalDataConfig):
        raise SystemExit("prepare_assets.py only supports RoboEvalDataConfig")
    H = cfg.model.action_horizon
    ds = factory.make_dataset(H)
    t0 = time.time()
    states, acts, pads = [], [], []
    for s, act, pad in ds.iter_chunks(a.num_chunks):
        states.append(s)
        acts.append(act)
        pads.append(pad)
    S, A, P = np.stack(states), np.stack(acts), np.stack(pads)
    print(f"sampled {len(S)} chunks in {time.time() - t0:.0f}s; state {S.shape} actions {A.shape}; pad frac {P.mean():.3f}")

    valid = ~P  # [n, H]
    Am = np.where(valid[..., None], A, np.nan)
    pt_std = np.nanstd(Am, axis=0)
    flat = A[valid]  # [m, D]
    q01, q99 = np.quantile(flat, 0.01, axis=0), np.quantile(flat, 0.99, axis=0)
    full = ~P.any(1)
    chol = correlation_cholesky(A[full])
    # min/max envelope + margin: rollout states outside a demo-only envelope would saturate the 256 state
    # bins. Joint velocities are spiky, so they use the [vel_q, 1 - vel_q] percentiles instead; values beyond
    # still reach the model unclipped through the sincos / continuous state encodings.
    s_lo, s_hi = S.min(0), S.max(0)
    v0, v1 = cfg.model.state_velocity_dims
    s_lo[v0:v1] = np.quantile(S[:, v0:v1], a.state_vel_quantile, axis=0)
    s_hi[v0:v1] = np.quantile(S[:, v0:v1], 1.0 - a.state_vel_quantile, axis=0)
    stats = {
        "state": normalize.NormStats(mean=S.mean(0), std=S.std(0),
                                     q01=s_lo - a.state_margin * (s_hi - s_lo),
                                     q99=s_hi + a.state_margin * (s_hi - s_lo)),
        "actions": normalize.NormStats(
            mean=flat.mean(0), std=flat.std(0), q01=q01, q99=q99,
            per_timestamp_std=pt_std,
            action_correlation_cholesky=chol,
        ),
    }
    np.set_printoptions(precision=4, suppress=True, linewidth=180)
    print("state min", S.min(0), "\nstate max", S.max(0))
    print("action std t=0 ", pt_std[0], "\naction std t=-1", pt_std[-1])
    asset_id = factory.assets.asset_id or factory.repo_id
    out = Path(factory.assets.assets_dir or cfg.assets_dirs) / asset_id
    normalize.save(out, stats)
    print(f"saved norm stats to {out}")

    from transformers import AutoProcessor

    ranges = cfg.model.get_fast_dim_ranges()
    dims = np.concatenate([np.arange(s, e) for s, e in ranges])
    sel = np.flatnonzero(full)[: a.fast_chunks]
    enc = A[sel][:, :, dims]
    lo, hi = q01[dims], q99[dims]
    enc = 2.0 * (np.clip(enc, lo, hi) - lo) / np.maximum(hi - lo, 1e-6) - 1.0
    base = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    t0 = time.time()
    tok = base.fit(list(enc), scale=a.fast_scale, vocab_size=cfg.model.fast_vocab_size,
                   time_horizon=enc.shape[1], action_dim=enc.shape[2])
    lens = np.array([len(tok(c[None])[0]) for c in enc[:2000]])
    print(f"FAST fit in {time.time() - t0:.0f}s; tokens/chunk mean {lens.mean():.1f} p99 {np.percentile(lens, 99):.0f} "
          f"max {lens.max()} (max_fast_tokens={cfg.model.max_fast_tokens})")
    tok_dir = out / "fast_tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(tok_dir)
    (tok_dir / "metadata.json").write_text(json.dumps({
        "vocab_size": cfg.model.fast_vocab_size, "scale": a.fast_scale, "encoded_dim_ranges": ranges,
        "action_horizon": H, "num_training_chunks": int(len(enc)),
        "tokens_mean": float(lens.mean()), "tokens_p99": float(np.percentile(lens, 99)), "tokens_max": int(lens.max()),
    }, indent=1))
    print(f"saved FAST tokenizer to {tok_dir}")


if __name__ == "__main__":
    main()
