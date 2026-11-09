"""Bit-exact CPU speedups for the RoboEval simulator, installed as runtime monkeypatches.

The RoboEval source stays untouched (the official evaluator fingerprints it). Each patch replaces a
per-step Python lookup with a cached or vectorised equivalent that returns exactly the same values:
static structure (which geoms belong to which body / collider list) is cached, anything that can
change during an episode (contacts, contype/conaffinity, poses) is still read live every call.
Replaying identical episodes with and without the patches gives bit-identical observations, rewards, infos and
metrics.

    from roboeval_solution.sim import speedups
    speedups.install()   # before or after building envs
"""
from __future__ import annotations

import weakref

import numpy as np

_COLLISION_MARGIN = 1e-8  # = roboeval.utils.physics_utils._DEFAULT_COLLISION_MARGIN
_installed = False

# physics -> {key: (elements, geom id set)}; elements are kept alive so their id() keys stay unique.
_geom_id_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _geom_id_set(physics, colliders) -> frozenset:
    els = tuple(c.mjcf for c in colliders)
    cache = _geom_id_cache.get(physics)
    if cache is None:
        cache = _geom_id_cache[physics] = {}
    key = tuple(map(id, els))
    hit = cache.get(key)
    if hit is not None and all(a is b for a, b in zip(hit[0], els)):
        return hit[1]
    ids = frozenset(int(i) for i in np.asarray(physics.bind(list(els)).element_id).ravel())
    cache[key] = (els, ids)
    return ids


def _as_list(colliders):
    from mojo.elements import Geom

    return [colliders] if isinstance(colliders, Geom) else colliders


def _close_contacts(physics):
    """(geom1, geom2) of contacts the original loop does not skip (`dist > margin` -> skip; NaN kept)."""
    contact = physics.data.contact
    if len(contact) == 0:
        return ()
    return [(a, b) for d, a, b in zip(contact.dist.tolist(), contact.geom1.tolist(), contact.geom2.tolist())
            if not d > _COLLISION_MARGIN]


def _touching(pairs, ids_1, ids_2) -> bool:
    for a, b in pairs:
        if (a in ids_1 and b in ids_2) or (b in ids_1 and a in ids_2):
            return True
    return False


# physics -> {id(body mjcf): (body mjcf, geom ids of the body subtree)}; the subtree is static after compile.
_body_geoms_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _body_geom_ids(physics, body_mjcf) -> tuple:
    per_phys = _body_geoms_cache.get(physics)
    if per_phys is None:
        per_phys = _body_geoms_cache[physics] = {}
    hit = per_phys.get(id(body_mjcf))
    if hit is not None and hit[0] is body_mjcf:
        return hit[1]
    geoms = body_mjcf.find_all("geom") or []
    ids = tuple(int(physics.bind(g).element_id) for g in geoms)
    per_phys[id(body_mjcf)] = (body_mjcf, ids)
    return ids


def gripper_is_holding_object(self, other) -> bool:
    """Same result as Gripper.is_holding_object: every pad body touches `other` with a collidable geom."""
    from roboeval.utils.physics_utils import get_colliders

    if not self._pad_bodies:
        return False
    physics = self._mojo.physics
    other_colliders = _as_list(get_colliders(other))
    if len(other_colliders) == 0:
        return _orig["gripper_is_holding_object"](self, other)
    contype, conaffinity = physics.model.geom_contype, physics.model.geom_conaffinity
    pad_sets = []
    for body in self._pad_bodies:
        # Geom.is_collidable(), read live (props can be disabled / enabled during an episode)
        ids = frozenset(g for g in _body_geom_ids(physics, body.mjcf) if contype[g] == 1 and conaffinity[g] == 1)
        if not ids:
            return _orig["gripper_is_holding_object"](self, other)
        pad_sets.append(ids)
    pairs = _close_contacts(physics)
    if not pairs:
        return False
    other_ids = _geom_id_set(physics, other_colliders)
    return all(_touching(pairs, ids, other_ids) for ids in pad_sets)


_orig: dict = {}


def install() -> None:
    global _installed
    if _installed:
        return
    from roboeval.robots import gripper

    # physics_utils.has_collided_collections is already cheap (dm_control caches binds); only the gripper
    # check, which re-walks the pad bodies' subtrees every call, is replaced.
    _orig["gripper_is_holding_object"] = gripper.Gripper.is_holding_object
    gripper.Gripper.is_holding_object = gripper_is_holding_object
    _installed = True


def set_render(env, on: bool) -> None:
    """Skip (False) / resume (True) camera rendering in env.step()/reset() observations.

    Rendering has no effect on the simulation, so an env that only needs images on some steps (the
    ones that feed a policy request) can skip the rest.
    """
    env._rbe_render = bool(on)


def _get_observation(self):
    obs = {}
    if self._observation_config.proprioception:
        obs |= self._get_proprioception_obs()
    if self._use_pixels and getattr(self, "_rbe_render", True):
        obs |= self._get_visual_obs()
    if self._observation_config.privileged_information:
        obs |= self._get_task_privileged_obs()
    return obs


def install_lazy_render() -> None:
    """RoboEvalEnv.get_observation honouring set_render(); default (no flag) = stock behaviour."""
    from roboeval.roboeval_env import RoboEvalEnv

    if RoboEvalEnv.get_observation is not _get_observation:
        RoboEvalEnv.get_observation = _get_observation


class _SharedGLContext:
    """Stand-in for mujoco's GLContext: every mujoco.Renderer of the process renders through ONE EGL
    context (each keeps its own MjrContext with its own buffers and model assets), so a process stepping
    several envs holds one GL context instead of one per env. The context lives as long as the process.

    mujoco.Renderer binds its offscreen framebuffer only at construction, while mjr_readPixels reads
    the renderer's own buffer: in a shared context a render after another renderer's would draw into
    the wrong framebuffer and read back a stale frame. install_shared_gl() therefore rebinds the
    renderer's buffer before every render (_render_own_buffer)."""

    _ctx = None

    def __init__(self, max_width, max_height):
        if _SharedGLContext._ctx is None:
            _SharedGLContext._ctx = _orig["GLContext"](max_width, max_height)

    def make_current(self):
        _SharedGLContext._ctx.make_current()

    def free(self):
        pass


def _render_own_buffer(self, *, out=None):
    """mujoco.Renderer.render with this renderer's offscreen buffer bound first (shared GL context)."""
    from mujoco import _enums, _render

    if self._gl_context and self._mjr_context is not None:
        self._gl_context.make_current()
        _render.mjr_setBuffer(_enums.mjtFramebuffer.mjFB_OFFSCREEN, self._mjr_context)
    return _orig["Renderer.render"](self, out=out)


def install_shared_gl() -> None:
    import mujoco
    from mujoco import gl_context

    if gl_context.GLContext is _SharedGLContext or gl_context.GLContext is None:
        return
    _orig["GLContext"] = gl_context.GLContext
    gl_context.GLContext = _SharedGLContext
    _orig["Renderer.render"] = mujoco.Renderer.render
    mujoco.Renderer.render = _render_own_buffer
