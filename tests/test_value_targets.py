"""Value-head targets (data/value_targets.py), the dataset that emits them, and the weight-loader re-init. CPU only.

    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_value_targets.py
"""
import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roboeval_solution.data import score_reward as sr  # noqa: E402
from roboeval_solution.data import value_targets as vt  # noqa: E402
from test_score_reward import synthetic_episode  # noqa: E402

from roboeval_solution.training.config import DATA_ROOT  # noqa: E402

DEMO_STORE = Path(DATA_ROOT) / "stores" / "demos" / "LiftPotPositionAndOrientation"
CAMS = ("head", "front", "external", "left_wrist", "right_wrist")


def _values(rng, fam="LiftPot", T=80, success=False, termination="budget"):
    arrays, ep = synthetic_episode(rng, fam=fam, T=T, success=success, termination=termination)
    ep = dict(ep, source="rollout", recording_ok=True)
    return arrays, ep, vt.episode_value_arrays(ep, arrays, 0, T)


def test_episode_value_arrays_layout():
    rng = np.random.default_rng(0)
    for fam, succ, term in (("LiftPot", False, "budget"), ("PackBox", True, "success"), ("CubeHandover", False, "fail")):
        T = 70
        arrays, ep, v = _values(rng, fam, T, succ, term)
        incr = sr.step_increments(arrays, ep)
        J = sr.STAGE_COUNT[fam]
        assert v["togo"].shape == (T + 1, 11) and v["togo"].dtype == np.float32
        np.testing.assert_allclose(v["togo"], sr.togo_targets(incr, fam), rtol=1e-6)
        assert np.all(v["togo"][T] == 0)  # terminal row
        assert np.all(np.diff(v["togo"], axis=0) <= 1e-6)  # to-go never increases
        np.testing.assert_array_equal(v["stage_mask"], np.arange(5) < J)
        reached = v["stage_first"] != sr.NOT_AHEAD
        np.testing.assert_array_equal(reached[:J], arrays["stage_flags"][-1, :J] > 0.5)
        assert not reached[J:].any()
        assert v["success"] == float(succ) and v["completion_T"] == (T if succ else 0)
        if term == "fail":  # the fail fine enters the to-go targets
            assert v["togo"][0, 0] == pytest.approx(min(ep["budget_steps"] / 20 / sr.BUDGET_SECONDS[fam], 5.0))


@pytest.mark.parametrize("success", [False, True])
def test_frame_targets_rollout(success):
    rng = np.random.default_rng(1)
    T = 50
    arrays, ep, v = _values(rng, "StackTwoBlocks", T, success, "success" if success else "budget")
    for t in (0, 7, T - 31, T - 30, T - 1):
        out = vt.frame_targets(v, t, T, is_weight=0.7)
        assert set(out) == set(vt.VALUE_TARGET_KEYS)
        tf = min(t + 30, T)
        assert out["success_target"] == out["success_future_target"] == float(success)
        np.testing.assert_array_equal(out["togo_target"], v["togo"][t])
        np.testing.assert_array_equal(out["togo_future_target"], v["togo"][t] - v["togo"][tf])
        assert out["togo_mask"].all() and out["togo_future_mask"].all()
        np.testing.assert_array_equal(out["stage_target"], sr.stage_frame_targets(v["stage_first"], t))
        np.testing.assert_array_equal(out["stage_mask"], v["stage_mask"])
        # ψ(t) = [success, stage_target x5, togo[t] x11]; masks: success always, stage j < J, togo all
        w, m = out["wm_vector_target"], out["wm_vector_mask"]
        assert w.shape == m.shape == (sr.WM_VECTOR_DIM,)
        assert w[0] == float(success) and m[0]
        np.testing.assert_array_equal(w[1:6], out["stage_target"])  # = the stage head's target (Q residual)
        np.testing.assert_array_equal(m[1:6], v["stage_mask"])
        np.testing.assert_array_equal(w[6:], v["togo"][t])
        assert m[6:].all()
        if success:
            assert out["completion_target"] == pytest.approx(t / T)
            assert out["completion_future_target"] == pytest.approx(tf / T)
        else:  # failed rollout: completion·success = 0
            assert out["completion_target"] == 0.0 and out["completion_future_target"] == 0.0
        assert out["completion_weight_scale"] == 1.0 and out["is_weight"] == np.float32(0.7)
    # near the end the chunk spends everything that is left: togo[t] - togo[T] with togo[T] = 0
    np.testing.assert_array_equal(vt.frame_targets(v, T - 1, T)["togo_future_target"], v["togo"][T - 1])


def test_masked_and_recording_ok():
    rng = np.random.default_rng(2)
    out = vt.frame_targets(None, 3, 10)
    assert np.isnan(out["success_target"]) and np.isnan(out["completion_target"])
    assert not any(out[k].any() for k in ("stage_mask", "togo_mask", "togo_future_mask",
                                          "wm_vector_mask"))
    for k in ("stage_target", "togo_target", "wm_vector_target"):
        assert np.all(np.isfinite(out[k]))  # masked slots are finite (0), never NaN
    arrays, ep = synthetic_episode(rng, T=40, success=True)
    ep = dict(ep, source="rollout")
    assert vt.episode_value_arrays(dict(ep, recording_ok=False), arrays, 0, 40) is None
    assert vt.episode_value_arrays(dict(ep, termination="error"), arrays, 0, 40) is None
    # without a usable recording only the success targets survive
    s = vt.frame_targets(vt.success_only_values(ep), 5, 40)
    assert s["success_target"] == 1.0 and s["success_future_target"] == 1.0
    assert s["wm_vector_mask"][0] and not s["wm_vector_mask"][1:].any() and s["wm_vector_target"][0] == 1.0
    assert np.isnan(s["completion_target"]) and not s["stage_mask"].any() and not s["togo_mask"].any()
    assert vt.success_only_values(dict(ep, termination="error")) is None


def test_demo_completion():
    L = 100
    reward = np.zeros(L, np.float32)
    reward[90:] = 1.0  # first hit at row 90: success after eval step 89 -> T = 90 (shift 1)
    v = vt.demo_values({"success": True}, reward, L, shift=1)
    assert v["completion_T"] == 90 and v["completion_scale"] == vt.DEMO_COMPLETION_SCALE
    o = vt.frame_targets(v, 0, L)
    assert o["completion_target"] == pytest.approx(1 / 90)  # row 0 = eval frame 1
    assert vt.frame_targets(v, 88, L)["completion_target"] == pytest.approx(89 / 90)  # last decision anchor
    assert vt.frame_targets(v, 99, L)["completion_target"] == 1.0
    assert o["completion_weight_scale"] == np.float32(0.5)
    assert np.isnan(o["success_target"]) and np.isnan(o["completion_future_target"])
    assert not o["wm_vector_mask"].any() and not o["stage_mask"].any() and not o["togo_mask"].any()
    assert vt.demo_values({"success": True}, np.zeros(L), L, 1)["completion_T"] == L  # no reward hit: full length
    assert vt.demo_values({"success": False}, reward, L, 1) is None


# ── dataset integration ──────────────────────────────────────────────────────
def _jpeg(rng):
    from roboeval_solution.data.store import encode_jpeg
    return encode_jpeg(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))


def write_rollout_store(root: Path, rng, specs, advantage=None):
    """specs: list of (family, T, success, termination, recording_ok) -> a rollout store with the recorded arrays."""
    from roboeval_solution.data.store import StoreWriter

    w = StoreWriter(root, cameras=list(CAMS), fps=20, info={"source": "rollout"})
    for fam, T, success, term, ok in specs:
        arrays, ep = synthetic_episode(rng, fam=fam, T=T, success=success, termination=term)
        arrays = {k: np.asarray(a, np.float64 if k == "ev_acc" else (np.uint8 if k == "stage_flags" else np.float32))
                  for k, a in arrays.items()}
        arrays.update(state=rng.normal(size=(T, 36)).astype(np.float32),
                      state_gripper=rng.random((T, 2)).astype(np.float32),
                      actions=(0.01 * rng.normal(size=(T, 14))).astype(np.float32))
        img = _jpeg(rng)
        info = dict(ep, variation="PositionAndOrientation", source="rollout", recording_ok=ok,
                    stage_flags_reset=[0] * 5)
        w.add_episode(arrays, {c: [img] * T for c in CAMS}, info)
    w.close()
    if advantage is not None:
        from roboeval_solution.data.store import write_array
        write_array(root, "advantage", advantage)
    return root


SPECS = [("LiftPot", 60, True, "success", True), ("LiftPot", 80, False, "budget", True),
         ("StackTwoBlocks", 50, False, "fail", True), ("LiftPot", 40, True, "success", False),
         ("LiftPot", 30, False, "error", True)]


def _dataset(paths_shares, **kw):
    from roboeval_solution.data.dataset import RoboEvalDataset, SamplingConfig, StoreSource

    sources = [StoreSource(paths=(str(p),), share=s, default_advantage=d, prioritized=pr, name=n)
               for p, s, d, pr, n in paths_shares]
    return RoboEvalDataset(sources, action_horizon=30, sampling=SamplingConfig(), use_value_targets=True, **kw)


def test_dataset_value_targets(tmp_path, caplog):
    rng = np.random.default_rng(3)
    root = write_rollout_store(tmp_path / "rollouts" / "w00", rng, SPECS)
    have_demo = (DEMO_STORE / "meta.json").exists()
    srcs = [(tmp_path / "rollouts", 0.5, None, False, "rollouts")]  # no advantage.npy: not prioritized
    if have_demo:
        srcs.append((DEMO_STORE, 0.5, 0.3, False, "demos"))
    with caplog.at_level(logging.INFO, logger="roboeval_solution.data.dataset"):
        ds = _dataset(srcs)
    log = caplog.text
    assert "'rollout_ok': 3" in log and "'rollout_no_recording': 1" in log and "'rollout_error': 1" in log
    assert "clipped sampling mass" in log
    ri = 0
    r = ds.readers[ri]
    for ei, (fam, T, success, term, ok) in enumerate(SPECS):
        st = r.episodes[ei]["start"]
        for t in (0, T // 2, T - 1):
            item = ds[int(ds._offsets[ri] + st + t)]
            assert item["stage_target"].shape == (5,) and item["wm_vector_target"].shape == (17,)
            if term == "error":
                assert np.isnan(item["success_target"]) and not item["wm_vector_mask"].any()
            elif not ok:
                assert item["success_target"] == float(success) and not item["togo_mask"].any()
                assert np.isnan(item["completion_target"])
            else:
                ref = vt.frame_targets(vt.episode_value_arrays(r.episodes[ei], r.arrays, st, T), t, T)
                for k in vt.VALUE_TARGET_KEYS:
                    if k != "is_weight":
                        np.testing.assert_array_equal(item[k], ref[k], err_msg=k)
                assert item["is_weight"] > 0
            assert item["task_id"] >= 0
    if have_demo:
        ri = 1
        r = ds.readers[ri]
        e = r.episodes[0]
        item = ds[int(ds._offsets[ri] + e["start"] + 3)]
        hit = np.flatnonzero(r.arrays["reward"][e["start"]: e["start"] + e["length"]] > 0)
        T = int(hit[0]) if len(hit) else e["length"]
        assert item["completion_target"] == pytest.approx(4 / T)
        assert item["completion_weight_scale"] == np.float32(0.5)
        assert np.isnan(item["success_target"]) and not item["stage_mask"].any()
        assert not item["togo_mask"].any() and not item["wm_vector_mask"].any()
        assert item["advantage"] == np.float32(0.3)


def test_nan_advantage_zero_weight(tmp_path):
    rng = np.random.default_rng(4)
    lens = [T for _, T, *_ in SPECS]
    adv = rng.normal(size=sum(lens)).astype(np.float32)
    starts = np.r_[0, np.cumsum(lens)]
    adv[starts[3]: starts[5]] = np.nan  # compute_advantages skips recording_ok False / error episodes
    write_rollout_store(tmp_path / "r" / "w00", rng, SPECS, advantage=adv)
    for prioritized in (True, False):
        ds = _dataset([(tmp_path / "r", 1.0, None, prioritized, "rollouts")])
        w = ds.weights
        assert np.all(w[starts[3]: starts[5]] == 0)
        assert w[: starts[3]].sum() > 0 and np.isclose(w.sum(), 1.0)
        assert np.all(ds._is_weight[starts[3]: starts[5]] == 0)


def test_prioritized_without_advantages_zero_weight(tmp_path, caplog):
    """A prioritized collect without advantage.npy must not train as A=0 (the positive token)."""
    rng = np.random.default_rng(5)
    n = sum(T for _, T, *_ in SPECS)
    write_rollout_store(tmp_path / "r" / "w00", rng, SPECS)
    write_rollout_store(tmp_path / "s" / "w00", rng, SPECS, advantage=np.zeros(n, np.float32))
    with caplog.at_level(logging.WARNING, logger="roboeval_solution.data.dataset"):
        ds = _dataset([(tmp_path / "r", 0.5, None, True, "rollouts_r"), (tmp_path / "s", 0.5, None, True, "rollouts_s")])
    assert "prioritized source without advantages" in caplog.text
    assert np.all(ds.weights[:n] == 0) and np.isclose(ds.weights[n:].sum(), 1.0)


def test_clipped_is_weights():
    from roboeval_solution.data.dataset import _clipped_is_weights

    p = np.r_[np.full(10, 0.05 / 10), np.full(990, 0.95 / 990), np.zeros(5)]
    w, clip, mass = _clipped_is_weights(p, (0.05, 20.0), 0.05)
    assert clip == (0.05, 20.0) and mass == 0  # 1/(N p) = 0.2 and ~1.04 over N = 1000 frames
    np.testing.assert_allclose(w[:10], 0.2, rtol=1e-6)
    assert np.all(w[-5:] == 0)
    p = np.r_[np.full(10, 0.9 / 10), np.full(990, 0.1 / 990)]  # 1/(N p) = 0.011 on 90 % of the mass
    w, clip, mass = _clipped_is_weights(p, (0.05, 20.0), 0.05)
    assert clip[0] < 0.011 and mass <= 0.05
    np.testing.assert_allclose(w[:10], 1.0 / (1000 * 0.09), rtol=1e-6)


def test_transform_path_to_observation(tmp_path):
    """Dataset item -> repack -> RoboEvalInputs -> normalise -> model transforms -> collate -> Observation keeps
    every value field (the RoboEvalInputs allowlist drops unknown keys silently)."""
    pytest.importorskip("jax")
    from roboeval_solution.training import config as C
    from roboeval_solution.training import data_loader as dl
    from roboeval_solution.models.observation import Observation

    cfg = C.get_config("pi05_roboeval")
    if not (Path(cfg.assets_dirs) / cfg.data.assets.asset_id / "norm_stats.json").exists():
        pytest.skip("norm stats not available")
    rng = np.random.default_rng(5)
    write_rollout_store(tmp_path / "r" / "w00", rng, SPECS[:3])
    from roboeval_solution.data.dataset import StoreSource
    data = C.RoboEvalDataConfig(repo_id=cfg.data.repo_id, assets=cfg.data.assets,
                                sources=(StoreSource(paths=(str(tmp_path / "r"),), name="rollouts"),))
    data_config = data.create(cfg.assets_dirs, cfg.model)
    ds = dl.transform_dataset(dl.create_dataset(data_config, cfg.model.action_horizon), data_config)
    batch = dl._collate_fn([ds[i] for i in (0, 70, 150)])
    obs = Observation.from_dict(batch)
    assert obs.stage_target.shape == (3, 5) and obs.stage_mask.dtype == bool
    assert obs.wm_vector_target.shape == (3, 17) and obs.wm_vector_mask.shape == (3, 17)
    assert obs.togo_target.shape == (3, 11) and obs.completion_weight_scale.shape == (3,)
    assert obs.success_future_target.shape == (3,) and obs.is_weight.shape == (3,)
    assert obs.wm_vector_mask[:, 0].all()


# ── weight loading (fresh runs) ──────────────────────────────────────────────
def _save_ckpt(path, tree):
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str(path), tree)
    return str(path)


def test_weight_loader_pi05_base(tmp_path):
    """pi0.5 base -> the pi0.5 modules (action projections cut to our dims, vocab table -> 1-row dummy); every
    PiModified-only module is left out (keeps the model's init); a missing pi0.5 param fails loudly."""
    pytest.importorskip("orbax.checkpoint")
    from roboeval_solution.training.weight_loaders import PiModifiedWeightLoader

    f32 = lambda *s: np.ones(s, np.float32)  # noqa: E731
    pi05 = {"PaliGemma": {"llm": {"embedder": {"input_embedding": f32(10, 4)}, "w": f32(3, 3)}},
            "action_in_proj": {"kernel": np.arange(32 * 4, dtype=np.float32).reshape(32, 4), "bias": f32(4)},
            "time_mlp_in": {"kernel": f32(4, 4)}}
    ref = {"PaliGemma": {"llm": {"embedder": {"input_embedding": f32(1, 4)}, "w": f32(3, 3)}},
           "action_in_proj": {"kernel": f32(14, 4), "bias": f32(4)},
           "time_mlp_in": {"kernel": f32(4, 4)},
           "success_head": {"kernel": f32(4, 1)}, "action_correlation_cholesky": f32(2, 2)}
    out = PiModifiedWeightLoader(_save_ckpt(tmp_path / "pi05", {"params": pi05})).load(ref)
    assert set(out) == {"PaliGemma", "action_in_proj", "time_mlp_in"}
    assert np.all(out["PaliGemma"]["llm"]["embedder"]["input_embedding"] == 0)
    assert out["PaliGemma"]["llm"]["embedder"]["input_embedding"].shape == (1, 4)
    np.testing.assert_array_equal(out["action_in_proj"]["kernel"], pi05["action_in_proj"]["kernel"][:14])
    bad = {**pi05, "time_mlp_in": {}}
    bad.pop("time_mlp_in")
    with pytest.raises(ValueError, match="time_mlp_in"):
        PiModifiedWeightLoader(_save_ckpt(tmp_path / "bad", {"params": bad})).load(ref)


def test_weight_loader_same_tree_pimodified(tmp_path):
    """A PiModified checkpoint (nnx 'value' leaves) loads the whole current tree; another tree fails loudly."""
    pytest.importorskip("orbax.checkpoint")
    from roboeval_solution.training.weight_loaders import PiModifiedWeightLoader

    ref = {"PaliGemma": {"w": np.zeros((3, 3), np.float32)}, "success_head": {"kernel": np.zeros((4, 1), np.float32)},
           "action_correlation_cholesky": np.zeros((2, 2), np.float32)}
    ckpt = {"params": {"PaliGemma": {"w": {"value": np.full((3, 3), 2.0, np.float32)}},
                       "success_head": {"kernel": {"value": np.full((4, 1), 3.0, np.float32)}},
                       "action_correlation_cholesky": {"value": np.eye(2, dtype=np.float32)}}}
    out = PiModifiedWeightLoader(_save_ckpt(tmp_path / "pm", ckpt)).load(ref)
    assert set(out) == {"PaliGemma", "success_head"}  # the norm-stat Intermediates are never loaded
    assert np.all(out["PaliGemma"]["w"] == 2) and np.all(out["success_head"]["kernel"] == 3)
    other = {"params": {"PaliGemma": {"w": {"value": np.zeros((3, 3), np.float32)}},
                        "success_head": {"kernel": {"value": np.zeros((4, 2), np.float32)}}}}
    with pytest.raises(ValueError, match="not a checkpoint of this model"):
        PiModifiedWeightLoader(_save_ckpt(tmp_path / "other", other)).load(ref)


def test_action_loss_constant_divisor():
    """Every valid (frame, step, dim) element weighs the same: a chunk with k valid steps carries k / H of a full
    one."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from roboeval_solution.models.pi_modified import masked_action_loss

    loss = jnp.ones((3, 30, 14)).at[2].set(2.0)
    pad = jnp.zeros((3, 30), bool).at[1, 10:].set(True).at[2, 1:].set(True)
    np.testing.assert_allclose(np.asarray(masked_action_loss(loss, pad)), [1.0, 10 / 30, 2.0 / 30], rtol=1e-6)


def test_collate_refuses_none():
    pytest.importorskip("jax")
    from roboeval_solution.training import data_loader as dl

    item = {"advantage": np.float32(1.0), "index": np.int64(3)}
    assert dl._collate_fn([item, dict(item)])["advantage"].shape == (2,)
    with pytest.raises(ValueError, match="advantage"):
        dl._collate_fn([item, {"advantage": None, "index": np.int64(4)}])


def test_demo_advantages(tmp_path):
    """A store without advantages trains at the source's default_advantage; fixed_advantage wins over it."""
    from roboeval_solution.data.dataset import RoboEvalDataset, SamplingConfig, StoreSource
    from roboeval_solution.data.store import StoreWriter

    rng = np.random.default_rng(5)
    root = tmp_path / "demos" / "w00"
    w = StoreWriter(root, cameras=list(CAMS), fps=20, info={"source": "demo"})
    for uuid in ("a", "b", "c"):
        T = 40
        w.add_episode({"state": rng.normal(size=(T, 36)).astype(np.float32),
                       "state_gripper": rng.random((T, 2)).astype(np.float32),
                       "actions": (0.05 * rng.normal(size=(T, 14))).astype(np.float32)},
                      {c: [_jpeg(rng)] * T for c in CAMS},
                      {"family": "StackTwoBlocks", "variation": "", "success": True, "source": "demo", "uuid": uuid})
    w.close()
    src = StoreSource(paths=(str(tmp_path / "demos"),), share=1.0, default_advantage=0.3, name="demos")
    for s, want in ((src, 0.3), (dataclasses.replace(src, fixed_advantage=-1.0), -1.0)):
        ds = RoboEvalDataset([s], action_horizon=10, sampling=SamplingConfig(static_min_weight=1.0))
        eps = ds.readers[0].episodes
        mass = [ds.weights[e["start"]: e["start"] + e["length"]].sum() for e in eps]
        assert np.allclose(mass, mass[0], rtol=0.05)  # same length, same family: equal sampling mass
        assert [float(ds[e["start"] + 5]["advantage"]) for e in eps] == pytest.approx([want] * 3)


def test_is_scripted_flag(tmp_path):
    """Dataset items mark scripted frames (scripted generations and recovery runs' scripted layer): the model's
    scripted state / velocity dropout keys on it."""
    from roboeval_solution.data.dataset import RoboEvalDataset, SamplingConfig, StoreSource
    from roboeval_solution.data.store import StoreWriter

    rng = np.random.default_rng(7)
    w = StoreWriter(tmp_path / "s" / "w00", cameras=list(CAMS), fps=20, info={"source": "rollout"})
    kinds = [("scripted", None), ("recovery", "retry"), ("recovery", "peak"), ("rollout", None)]
    for src, scen in kinds:
        T = 30
        info = {"family": "LiftPot", "variation": "PositionAndOrientation", "success": True, "source": src}
        if scen:
            info["scenario"] = scen
        w.add_episode({"state": rng.normal(size=(T, 36)).astype(np.float32),
                       "state_gripper": rng.random((T, 2)).astype(np.float32),
                       "actions": (0.01 * rng.normal(size=(T, 14))).astype(np.float32)},
                      {c: [_jpeg(rng)] * T for c in CAMS}, info)
    w.close()
    ds = RoboEvalDataset([StoreSource(paths=(str(tmp_path / "s"),), share=1.0, name="mix")], action_horizon=10,
                         sampling=SamplingConfig(static_min_weight=1.0))
    flags = [float(ds[e["start"] + 3]["is_scripted"]) for e in ds.readers[0].episodes]
    assert flags == [1.0, 1.0, 0.0, 0.0]
