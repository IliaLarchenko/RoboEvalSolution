"""Training dataset over rbe stores (demos + rollouts), with weighted sampling.

Each item is a dict in the "raw LeRobot-like" layout consumed by the repack
transform of ``RoboEvalDataConfig``:

    observation.images.<key>    uint8 [H, W, 3] per camera_map entry
    observation.state           float32 [S]
    action                      float32 [H_act, 14]  (model representation)
    action_is_pad               bool [H_act]
    task_id                     int32 task family id
    episode_step                float32 env steps since the reset (aux heads' time input)
    is_demo, is_scripted        float32 0 / 1 source flags
    advantage                   float32
    value-head targets          data/value_targets.py VALUE_TARGET_KEYS (use_value_targets)
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from roboeval_solution.data import action_repr as ar
from roboeval_solution.data import value_targets as vt
from roboeval_solution.data.store import StoreReader, list_stores
from roboeval_solution.data.tasks import TASK_TO_ID

logger = logging.getLogger(__name__)

# model camera key -> store camera name (all five RoboEval cameras)
DEFAULT_CAMERA_MAP = (("head_rgb", "head"), ("front_rgb", "front"), ("external_rgb", "external"),
                      ("left_rgb", "left_wrist"), ("right_rgb", "right_wrist"))


# per-frame store arrays __getitem__ reads: all a loader worker needs
ITEM_ARRAYS = ("actions", "state", "state_gripper", "episode_index", "advantage")


@dataclasses.dataclass(frozen=True)
class StoreSource:
    """A group of stores sampled with one share of the batch."""

    paths: tuple[str, ...]
    share: float = 1.0
    # Only use episodes with success == True (rollout stores carry failures too).
    success_only: bool = False
    # Advantage injected when the store has no 'advantage' array (demos, scripted generations).
    default_advantage: float | None = None
    # Every frame of the source at this advantage (over the store's advantage array and the default), e.g. -1 for
    # demos kept only as the negative side of classifier-free guidance.
    fixed_advantage: float | None = None
    # Weight frames by exp(clip(advantage)) when the store has advantages.
    prioritized: bool = False
    name: str = ""
    # Per-frame weights times this factor (the flywheel puts the demos' frames x sample_share here).
    weight_scale: float = 1.0


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    # Per-family weight ∝ (frames_family)^-task_balance_power within each source.
    # 0 = natural frame distribution, 1 = every family gets equal mass.
    task_balance_power: float = 0.5
    # Extra multiplier per variation name ("" = base variant).
    variation_weights: tuple[tuple[str, float], ...] = (("PositionAndOrientation", 2.0),)
    advantage_clip: tuple[float, float] = (-2.0, 2.0)
    # Static-frame reweight (from our BEHAVIOR-1K solution): anchors whose next `static_horizon` commanded motion is
    # tiny are down-weighted (weight 0 below `static_lo`, linear to 1 at `static_hi`; the
    # measure is max over the window of the cumulative end-effector translation (m) of both
    # arms + 0.1 m/rad x rotation, gripper toggles count as motion).
    static_horizon: int = 30
    static_lo: float = 0.001
    static_hi: float = 0.004
    static_min_weight: float = 0.0
    # Aux-loss importance weights 1/(N p) are clipped to this range; it is widened (x2 per side) until at most
    # `is_clip_max_mass` of the sampling mass is clipped (a clipped value target biases the heads).
    is_weight_clip: tuple[float, float] = (0.05, 20.0)
    is_clip_max_mass: float = 0.05


def _obs_action_shift(r: StoreReader) -> int:
    """1 for stores converted from the released demos (post-action observations), else 0."""
    return 1 if r.meta.get("info", {}).get("source") == "roboeval_ee_delta_20hz_scaled" else 0


def _future_motion(actions: np.ndarray, episode_index: np.ndarray, horizon: int) -> np.ndarray:
    """Per frame: max over the next `horizon` actions (same episode) of the cumulative commanded
    motion |sum dpos_L| + |sum dpos_R| + 0.1 (|sum drot_L| + |sum drot_R|); inf if a gripper toggles."""
    n = len(actions)
    a = actions.astype(np.float64)
    out = np.zeros(n)
    C = np.concatenate([np.zeros((1, 12)), np.cumsum(a[:, :12], axis=0)])  # C[t] = sum of a[:t]
    g = a[:, 12:14] > 0.5
    for k in range(1, horizon + 1):
        idx = np.minimum(np.arange(n) + k, n)
        same = np.concatenate([episode_index, [-1]])[np.minimum(np.arange(n) + k - 1, n)] == episode_index
        d = C[idx] - C[:n]
        m = (np.linalg.norm(d[:, 0:3], axis=1) + np.linalg.norm(d[:, 6:9], axis=1)
             + 0.1 * (np.linalg.norm(d[:, 3:6], axis=1) + np.linalg.norm(d[:, 9:12], axis=1)))
        toggled = np.any(g[np.minimum(np.arange(n) + k - 1, n - 1)] != g, axis=1)
        m = np.where(toggled, np.inf, m)
        out = np.where(same, np.maximum(out, m), out)
    return out


def _clipped_is_weights(p: np.ndarray, clip: tuple[float, float], max_mass: float, max_widen: int = 6
                        ) -> tuple[np.ndarray, tuple[float, float], float]:
    """1/(N p) over frames with p > 0, clipped; the clip is widened until <= max_mass of the sampling mass
    is clipped. Returns (weights, final clip, clipped mass)."""
    pos = p > 0
    raw = np.zeros_like(p)
    raw[pos] = 1.0 / (pos.sum() * p[pos])
    lo, hi = clip
    for i in range(max_widen + 1):
        mass = float(p[pos & ((raw < lo) | (raw > hi))].sum())
        if mass <= max_mass or i == max_widen:
            break
        lo, hi = lo / 2.0, hi * 2.0
    return np.where(pos, np.clip(raw, lo, hi), 0.0).astype(np.float32), (lo, hi), mass


class RoboEvalDataset:
    def __init__(
        self,
        sources: Sequence[StoreSource],
        action_horizon: int,
        camera_map: Sequence[tuple[str, str]] = DEFAULT_CAMERA_MAP,
        sampling: SamplingConfig = SamplingConfig(),
        use_value_targets: bool = False,
    ):
        self.action_horizon = action_horizon
        self.camera_map = tuple(camera_map)
        self.readers: List[StoreReader] = []
        self.reader_source: List[int] = []
        self.sources = list(sources)
        for si, src in enumerate(self.sources):
            roots = list_stores(src.paths)
            if not roots:
                raise FileNotFoundError(f"source {src.name or si}: no stores under {src.paths}")
            for r in roots:
                reader = StoreReader(r)
                if len(reader) == 0:  # an empty store (a worker that kept no episode) has no arrays at all
                    continue
                self.readers.append(reader)
                self.reader_source.append(si)
        # Global frame index -> (reader, local frame).
        sizes = np.array([len(r) for r in self.readers], np.int64)
        self._offsets = np.concatenate([[0], np.cumsum(sizes)])
        self._total = int(self._offsets[-1])
        # Observation/action alignment per store. The released demos were recorded as
        # (obs AFTER a_t, a_t) (DemoConverter.create_demo_in_new_env stores env.step(a_t)'s obs with
        # a_t), i.e. row t's obs is the eval obs_{t+1}: the chunk for anchor row t starts at row
        # t + 1. Our rollout stores are eval-aligned (obs before the action): shift 0.
        self._shift = [_obs_action_shift(r) for r in self.readers]
        # Per-reader episode lookup arrays.
        self._ep_start = [np.array([e["start"] for e in r.episodes], np.int64) for r in self.readers]
        self._ep_len = [np.array([e["length"] for e in r.episodes], np.int64) for r in self.readers]
        self.weights = self._compute_weights(sampling)
        self.use_value_targets = use_value_targets
        # Per reader, per episode: value-target dict for vt.frame_targets (None = all masked).
        self._values: List[List[Optional[dict]]] = []
        if use_value_targets:
            counts: Dict[str, int] = {}
            for r, shift in zip(self.readers, self._shift):
                self._values.append(self._episode_values(r, shift, counts))
            bad = counts.get("rollout_no_recording", 0) + counts.get("rollout_error", 0)
            (logger.warning if bad else logger.info)(
                "value targets per episode: %s%s", dict(sorted(counts.items())),
                " (rollouts without a usable recording keep success targets only; error episodes are fully "
                "masked)" if bad else "")
            # Importance weights making aux losses unbiased w.r.t. uniform frames despite
            # prioritised / share-based sampling: E_p[1 / (N p)] = 1.
            self._is_weight, clip, mass = _clipped_is_weights(self.weights, sampling.is_weight_clip,
                                                              sampling.is_clip_max_mass)
            msg = "aux IS weights 1/(N p): clip [%.4g, %.4g], clipped sampling mass %.2f%%"
            if clip != tuple(sampling.is_weight_clip):
                logger.warning(msg + " (widened from %s: > %.0f%% of the mass was clipped)", *clip, 100 * mass,
                               sampling.is_weight_clip, 100 * sampling.is_clip_max_mass)
            else:
                logger.info(msg, *clip, 100 * mass)
        logger.info("RoboEvalDataset: %d stores, %d frames, %d episodes", len(self.readers), self._total,
                    sum(len(r.episodes) for r in self.readers))

    @staticmethod
    def _episode_values(r: StoreReader, shift: int, counts: Dict[str, int]) -> List[Optional[dict]]:
        """Value targets per episode of one store (data/value_targets.py): full targets for recorded rollout
        episodes, success only for rollouts without a usable recording, completion only for successful demos.
        `counts` accumulates episodes per kind."""
        out: List[Optional[dict]] = []
        reward = r.arrays.get("reward")
        for e in r.episodes:
            st, n = int(e["start"]), int(e["length"])
            # branched hard-case episodes (sim/recovery.py) and the privileged scripted policy's episodes
            # (rollout.py --scripted): BC only, no aux (not the served policy's behaviour)
            if e.get("source") in ("recovery", "scripted"):
                v, kind = None, e["source"]
                counts[kind] = counts.get(kind, 0) + 1
                out.append(v)
                continue
            if e.get("source") == "rollout":
                v = vt.episode_value_arrays(e, r.arrays, st, n)
                kind = "rollout_ok"
                if v is None:
                    v = vt.success_only_values(e)
                    kind = "rollout_error" if e.get("termination") == "error" else "rollout_no_recording"
            else:
                v = vt.demo_values(e, reward[st: st + n], n, shift)
                kind = "demo_completion" if v is not None else "demo_masked"
            counts[kind] = counts.get(kind, 0) + 1
            out.append(v)
        return out

    # ── sampling weights ────────────────────────────────────────────────────
    def _compute_weights(self, cfg: SamplingConfig) -> np.ndarray:
        var_w = dict(cfg.variation_weights)
        w = np.zeros(self._total, np.float64)
        for si in range(len(self.sources)):
            src = self.sources[si]
            ridx = [i for i, s in enumerate(self.reader_source) if s == si]
            # frame weights inside the source before share normalisation
            fam_frames: Dict[str, float] = {}
            parts = []
            for i in ridx:
                r = self.readers[i]
                fw = np.zeros(len(r), np.float64)
                shift = _obs_action_shift(r)
                motion = _future_motion(r.arrays["actions"], r.arrays["episode_index"], cfg.static_horizon)
                if shift:  # the anchor's chunk starts `shift` rows later
                    motion = np.concatenate([motion[shift:], np.zeros(shift)])
                static_w = np.clip((motion - cfg.static_lo) / max(cfg.static_hi - cfg.static_lo, 1e-9),
                                   cfg.static_min_weight, 1.0)
                # As in our BEHAVIOR-1K solution, frames after the first success frame (reward > 0) are never chunk
                # anchors; the completion frame stays, and earlier chunks keep their real (post-success) targets.
                rew = r.arrays.get("reward")
                for e in r.episodes:
                    if src.success_only and not e.get("success", False):
                        continue
                    st, n = e["start"], e["length"]
                    sw = static_w[st: st + n].copy()
                    if e.get("success", False):  # the approach to success is never "static"
                        sw[max(0, n - cfg.static_horizon):] = 1.0
                    fw[st: st + n] = var_w.get(e.get("variation", ""), 1.0) * sw
                    if shift:  # anchors without a next action
                        fw[st + n - shift: st + n] = 0.0
                    if rew is not None:
                        # Demo reward at row r is the reward after a_{r-1} (shifted by the converter):
                        # the success state is row r-1, the eval policy's last decision is anchor r-2.
                        hit = np.flatnonzero(rew[st: st + n] > 0)
                        if len(hit):
                            last_anchor = hit[0] - 2 if shift else hit[0]
                            fw[st + max(last_anchor + 1, 0): st + n] = 0.0
                if "advantage" in r.arrays:
                    adv = r.arrays["advantage"].astype(np.float64)
                    # compute_advantages.py writes NaN for episodes it skips (recording not ok, error):
                    # never sample them (a 0 advantage would read as positive to RECAP).
                    bad = ~np.isfinite(adv)
                    if bad.any():
                        fw[bad] = 0.0
                        logger.info("%s: %d frames with non-finite advantage get sampling weight 0",
                                    r.root.name, int(bad.sum()))
                    if src.prioritized:
                        fw *= np.exp(np.clip(np.where(bad, 0.0, adv), *cfg.advantage_clip))
                elif src.prioritized:
                    # a rollout store without advantage.npy would train as A=0 (the positive token) with
                    # uniform weights: exclude it until compute_advantages.py has scored it
                    logger.warning("!!! %s: prioritized source without advantages: its %d frames get sampling "
                                   "weight 0 (run compute_advantages.py over the pool)", r.root, len(fw))
                    fw[:] = 0.0
                for e in r.episodes:
                    fam_frames[e["family"]] = fam_frames.get(e["family"], 0.0) + float(
                        fw[e["start"]: e["start"] + e["length"]].sum())
                parts.append((i, fw, r))
            for i, fw, r in parts:
                for e in r.episodes:
                    m = fam_frames.get(e["family"], 0.0)
                    if m > 0:
                        fw[e["start"]: e["start"] + e["length"]] *= m ** (-cfg.task_balance_power)
            total, share = sum(fw.sum() for _, fw, _ in parts), src.share
            if total <= 0:
                logger.warning("source %s has zero sampling mass", src.name or si)
                continue
            scale = src.weight_scale
            for i, fw, _ in parts:
                w[self._offsets[i]: self._offsets[i + 1]] = fw / total * share * scale
        w /= w.sum()
        # Report the realised family mix.
        mix: Dict[str, float] = {}
        for i, r in enumerate(self.readers):
            for e in r.episodes:
                s = self._offsets[i] + e["start"]
                mix[e["family"]] = mix.get(e["family"], 0.0) + float(w[s: s + e["length"]].sum())
        logger.info("sampling mass per family: %s", {k: round(v, 3) for k, v in sorted(mix.items())})
        kinds: Dict[str, float] = {}
        for i, r in enumerate(self.readers):
            k = (self.sources[self.reader_source[i]].name or "demos").partition("_")[0]
            kinds[k] = kinds.get(k, 0.0) + float(w[self._offsets[i]: self._offsets[i + 1]].sum())
        logger.info("sampling mass per source kind: %s", {k: round(v, 3) for k, v in sorted(kinds.items())})
        return w.astype(np.float64)

    def __getstate__(self):
        """The copy each loader worker gets (spawn pickles the dataset once per worker): the readers keep only the
        per-frame arrays __getitem__ reads (ITEM_ARRAYS; the others only feed the weights and value targets built
        here), and no sampling weights (the sampler runs in the main process)."""
        d = dict(self.__dict__)
        d["readers"] = [r.with_arrays(ITEM_ARRAYS) for r in self.readers]
        d["weights"] = None
        return d

    # ── item access ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self._total

    def locate(self, index: int) -> tuple[int, int, int]:
        """global frame -> (reader, local frame, local episode)."""
        ri = int(np.searchsorted(self._offsets, index, side="right") - 1)
        fi = int(index - self._offsets[ri])
        ei = int(self.readers[ri].arrays["episode_index"][fi])
        return ri, fi, ei

    def action_chunk(self, ri: int, fi: int, ei: int) -> tuple[np.ndarray, np.ndarray]:
        r = self.readers[ri]
        start, length = int(self._ep_start[ri][ei]), int(self._ep_len[ri][ei])
        end = start + length
        H = self.action_horizon
        a0 = min(fi + self._shift[ri], end)  # first action of this anchor's chunk
        hi = min(a0 + H, end)
        raw = np.zeros((H, ar.ACTION_DIM), np.float32)
        raw[: hi - a0] = r.arrays["actions"][a0:hi]
        pad = np.zeros(H, bool)
        if hi - a0 < H:
            pad[hi - a0:] = True
            raw[hi - a0:, ar.GRIP] = r.arrays["actions"][max(hi - 1, start), ar.GRIP]  # hold pose + gripper
        return ar.to_model(raw), pad

    def state(self, ri: int, fi: int) -> np.ndarray:
        r = self.readers[ri]
        return ar.make_state(r.arrays["state"][fi], r.arrays["state_gripper"][fi])

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        ri, fi, ei = self.locate(index)
        r = self.readers[ri]
        ep = r.episodes[ei]
        actions, pad = self.action_chunk(ri, fi, ei)
        st, n = int(ep["start"]), int(ep["length"])
        item = {
            # env steps since the reset at this row's observation: a branched recovery episode starts after its
            # replayed prefix (t0); demo rows hold the obs after their action (+ shift)
            "episode_step": np.float32(fi - st + self._shift[ri] + int(ep.get("t0", 0))),
            "observation.state": self.state(ri, fi),
            "action": actions,
            "action_is_pad": pad,
            "task_id": np.int32(TASK_TO_ID[ep["family"]]),
            # 1 for demonstration frames, 0 for rollouts (per-source loss logging)
            "is_demo": np.float32(self.sources[self.reader_source[ri]].name == "demos" or ep.get("source") == "demo"),
            # scripted actions: a scripted generation, or a recovery run's scripted continuation (taking over the
            # failed state: retry, or steering back onto its path from the deviation: track)
            "is_scripted": np.float32(ep.get("source") == "scripted" or (
                ep.get("source") == "recovery" and ep.get("scenario") in ("retry", "track"))),
        }
        for model_key, cam in self.camera_map:
            item[f"observation.images.{model_key}"] = r.image(cam, fi)
        if self.use_value_targets:
            item.update(vt.frame_targets(self._values[ri][ei], fi - st, n, is_weight=float(self._is_weight[index])))
        src = self.sources[self.reader_source[ri]]
        if src.fixed_advantage is not None:
            adv = float(src.fixed_advantage)
        elif "advantage" in r.arrays:
            adv = float(r.arrays["advantage"][fi])
        else:
            adv = src.default_advantage if src.default_advantage is not None else 0.0
        item["advantage"] = np.float32(adv if np.isfinite(adv) else 0.0)
        return item

    def iter_chunks(self, n: int, seed: int = 0):
        """Sample n (state, action chunk, pad) triples by the training distribution (for stats)."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(self._total, size=min(n, self._total), replace=False, p=self.weights)
        for index in idx:
            ri, fi, ei = self.locate(int(index))
            a, pad = self.action_chunk(ri, fi, ei)
            yield self.state(ri, fi), a, pad
