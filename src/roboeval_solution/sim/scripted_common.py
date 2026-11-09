"""Helpers shared by the privileged scripted policies (sim/scripted_*.py): measured and commanded hand poses, body
poses, the env's holding check, the per-episode parameter draw and the velocity servo most policies use."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def draw_params(randomize: dict, seed: int) -> dict:
    """Per-episode uniform draws of a policy's RANDOMIZE ranges (ints are drawn as ints, both ends included)."""
    rng = np.random.default_rng(seed)
    return {k: int(rng.integers(lo, hi + 1)) if isinstance(lo, int) else float(rng.uniform(lo, hi))
            for k, (lo, hi) in randomize.items()}


def _site_ids(env):
    ids = getattr(env, "_rbe_scripted_sites", None)
    if ids is None:
        ph = env._mojo.physics
        ids = [ph.model.name2id(n.replace("\\", "/"), "site") for n in env._robot.ik_config.arm_sites]
        env._rbe_scripted_sites = ids
    return ids


def ee_sites(env) -> np.ndarray:
    """[2, 6] measured IK-site poses (world pos + rotvec), arm 0 = left."""
    d = env._mojo.physics.data
    ids = _site_ids(env)
    rv = R.from_matrix(np.asarray(d.site_xmat[ids]).reshape(-1, 3, 3)).as_rotvec()
    return np.concatenate([np.asarray(d.site_xpos[ids]), rv], 1)


def cmd_pose(env) -> np.ndarray:
    """[2, 6] commanded poses: FK of the actuator targets, robot-base frame (what ee-delta actions compose onto)."""
    r = env.robot
    return np.asarray(r.forward_kinematics(joint_positions=r.get_initial_joint_positions()), float).reshape(2, 6)


def pose(body):
    """(world position, rotation) of a prop body."""
    q = np.asarray(body.get_quaternion(), float)  # wxyz
    return np.asarray(body.get_position(), float), R.from_quat([q[1], q[2], q[3], q[0]])


def holding(env, obj, arm: int) -> bool:
    """The env's own check that hand `arm` (0 = left) holds `obj`."""
    from roboeval.const import HandSide

    return env.robot.is_gripper_holding_object(obj, HandSide.LEFT if arm == 0 else HandSide.RIGHT)


def opening(env, arm: int) -> float:
    """Gripper `arm`'s opening (0 closed .. 1 open) as the observation's proprioception_grippers reports it, unrounded
    (the observation rounds it to 0.1)."""
    from roboeval.const import HandSide

    g = env.robot.grippers[HandSide.LEFT if arm == 0 else HandSide.RIGHT]
    ph = env._mojo.physics
    return float(np.mean([np.interp(ph.bind(j).qpos.item(), ph.bind(j).range, g._config.range)
                          for j in g._actuated_joints]))


def pad_mid(env, arm: int) -> np.ndarray:
    """World position of the midpoint between gripper `arm`'s two fingertip pads (their collision boxes)."""
    from roboeval.const import HandSide

    ids = getattr(env, "_rbe_pad_boxes", None)
    if ids is None:
        ph = env._mojo.physics
        ids = {}
        for a, side in ((0, HandSide.LEFT), (1, HandSide.RIGHT)):
            gs = [ph.bind(g._mjcf if hasattr(g, "_mjcf") else g.mjcf).element_id
                  for b in env.robot.grippers[side]._pad_bodies for g in b.geoms if g.is_collidable()]
            ids[a] = [i for i in gs if ph.model.geom_type[i] == 6]  # mjGEOM_BOX: the fingertip pads
        env._rbe_pad_boxes = ids
    return np.asarray(env._mojo.physics.data.geom_xpos[ids[arm]], float).mean(0)


def smoothstep(x: float, lo: float, hi: float) -> float:
    """0 below lo, 1 above hi, a smooth cubic in between."""
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def down(angle: float) -> R:
    """Top-down gripper rotation with the finger axis at `angle` (world yaw)."""
    y = np.array([np.cos(angle), np.sin(angle), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return R.from_matrix(np.stack([np.cross(y, z), y, z], 1))


def servo(p: dict, v: np.ndarray, arm: int, cmd: np.ndarray, target, rt: R, vmax=None, vmax_xy=None,
          stop_cap: bool = False):
    """One step of arm `arm` toward (target, rt) on its commanded pose: velocity kp * error, |v| <= vmax (default
    p["vmax"]; optionally |v_xy| <= vmax_xy, and with stop_cap no faster than it can stop within the remaining
    distance), speed change <= amax; rotation <= max_dr. Updates v[arm] (the commanded velocity state) and returns
    (dp, dr)."""
    vmax = p["vmax"] if vmax is None else vmax
    v_des = p["kp"] * (target - cmd[arm, :3])
    if vmax_xy is not None:  # a lower cap on the horizontal speed only
        n = np.linalg.norm(v_des[:2])
        if n > vmax_xy:
            v_des[:2] *= vmax_xy / n
    if stop_cap:  # no overshoot at the end of a move
        vmax = min(vmax, float(np.sqrt(2.0 * p["amax"] * np.linalg.norm(target - cmd[arm, :3]))))
    n = np.linalg.norm(v_des)
    if n > vmax:
        v_des *= vmax / n
    dv = v_des - v[arm]
    n = np.linalg.norm(dv)
    if n > p["amax"]:
        dv *= p["amax"] / n
    v[arm] = v[arm] + dv
    dr = (rt * R.from_rotvec(cmd[arm, 3:]).inv()).as_rotvec()
    n = np.linalg.norm(dr)
    if n > p["max_dr"]:
        dr *= p["max_dr"] / n
    return v[arm].copy(), dr
