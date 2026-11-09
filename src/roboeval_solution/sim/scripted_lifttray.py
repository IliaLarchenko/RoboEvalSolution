"""Scripted LiftTray policy with privileged tray pose (rollout.py --scripted lifttray).

Both hands reach over the tray's two end rims together and come straight down onto them at the same time (left hand:
the rim at larger world y), fingers across the rim thickness (tray local x; of the parallel gripper's two symmetric
yaws the one with the finger axis toward world +y, in every episode), site 0.163 above the tray origin
as in the demos / policy; close, and once both hold, carry the tray along one path: slide it in toward the body and
toward the arms' midline while turning it square (long axis along world y), rising over the later part, until it is
clear -> success (both holding, off the table, lifted >= 0.05 by the env's measure, i.e. ~2.3 cm).

Why this path: the tray is heavy (34 kg) for the arms (shoulder / elbow 87 Nm, wrist 12 Nm); at arm's length the arms
cannot lift it, and when its long axis is turned the far hand is the weaker one, so the tray lifted unsquared tilts
toward the far hand (hand height difference = the bgvd metric). Squared and pulled in, both arms carry it alike.

Motion: a velocity command per arm toward its target (gain kp, plus the path's feed-forward step in the lift) on the
commanded pose (FK of the actuator targets + an FK->world offset measured at t=0), with one common scale for both arms
(|v| <= vmax, |dv| <= amax per step) so the hands move in step; rotation <= max_dr per step.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim.scripted_common import cmd_pose, down, ee_sites, holding, pose, smoothstep

MID_Y = -0.3  # midline between the arm bases (hands start at y 0 / -0.6)
RIM_X = 0.265  # rim centre along the tray's local x (its long axis)
GRASP_Y = 0.163  # site above the tray origin along the tray's local y (world up) at the grasp (demos / policy)

DEFAULTS = dict(center_frac=0.65, pull_dx=0.27, pre_dz=0.07, lift_dz=0.29, rim_z=0.01, kp=0.35, vmax=0.018,
                amax=0.004, max_dr=0.08, grasp_tol=0.006, via_tol=0.02, close_steps=12, square_frac=1.0, rise_on=0.3,
                lift_v=0.012)
RANDOMIZE = dict(center_frac=(0.5, 0.8), pull_dx=(0.25, 0.30), lift_dz=(0.26, 0.32), rim_z=(-0.008, 0.015),
                 pre_dz=(0.05, 0.09), kp=(0.3, 0.4), vmax=(0.016, 0.020), amax=(0.0035, 0.0045), square_frac=(0.85, 1.0),
                 rise_on=(0.2, 0.4), lift_v=(0.011, 0.013))


def _minjerk(tau: float) -> float:
    """Minimum-jerk progress 0 -> 1 over tau in [0, 1] (zero speed and acceleration at both ends)."""
    tau = min(1.0, max(0.0, tau))
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


class ScriptedLiftTray:
    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.v = np.zeros((2, 3))
        self.phase = {0: "pre", 1: "pre"}
        self.k = {0: 0, 1: 0}
        self.events: set[str] = set()
        self.offset = None

    def _servo(self, cmd, targets, ff, rts):
        """Both arms: velocity toward the target (gain kp, plus the path's feed-forward step), one common scale for
        the speed (|v| <= vmax) and the speed change (|dv| <= amax), so both hands cover the same fraction of their way
        each step and arrive together; rotation <= max_dr per step."""
        p = self.p
        v_des = np.stack([ff[j] + p["kp"] * (targets[j] - cmd[j, :3]) for j in (0, 1)])
        n = np.linalg.norm(v_des, axis=1).max()
        if n > p["vmax"]:
            v_des *= p["vmax"] / n
        dv = v_des - self.v
        n = np.linalg.norm(dv, axis=1).max()
        if n > p["amax"]:
            dv *= p["amax"] / n
        self.v = self.v + dv
        out = []
        for j in (0, 1):
            dr = (rts[j] * R.from_rotvec(cmd[j, 3:]).inv()).as_rotvec()
            n = np.linalg.norm(dr)
            if n > p["max_dr"]:
                dr *= p["max_dr"] / n
            out.append((self.v[j].copy(), dr))
        return out

    def _setup(self, env):
        tray = env.breakfast_tray
        p0, rt = pose(tray.body)
        ends = [rt.apply([s * RIM_X, GRASP_Y, self.p["rim_z"]]) for s in (1.0, -1.0)]  # offsets from the origin
        hi = int(np.argmax([e[1] for e in ends]))
        # left hand: the rim at larger world y
        self.rim_local = {0: rt.inv().apply(ends[hi]), 1: rt.inv().apply(ends[1 - hi])}
        ax = rt.apply([1.0, 0.0, 0.0])  # finger axis: across the rim thickness, horizontal
        a0 = np.arctan2(ax[1], ax[0])
        # of the two symmetric yaws, the one whose finger axis points toward world +y (both hands, every episode)
        f = max([a0, a0 + np.pi], key=np.sin)
        self.finger = {0: f, 1: f}

    def _lift_plan(self, cur, p0):
        """The tray's path once both hands hold it: slide it in toward the body and sideways onto the arms' midline
        while turning it square (long axis along world y, both hands at the same reach), rising over the last part.
        Hand targets are the grasp points carried rigidly with the tray, so the two hands never fight each other."""
        p = self.p
        self.lift_from = {j: cur[j, :3].copy() for j in (0, 1)}
        self.c0 = p0[:2].copy()
        d = self.lift_from[0][:2] - self.lift_from[1][:2]  # right hand -> left hand
        self.dpsi = p["square_frac"] * ((np.pi / 2 - np.arctan2(d[1], d[0]) + np.pi) % (2 * np.pi) - np.pi)
        self.dc = np.array([-p["pull_dx"], p["center_frac"] * (MID_Y - self.c0[1])])
        ends = [self._lift_target(j, 1.0)[0] for j in (0, 1)]
        dist = max(np.linalg.norm(ends[j] - self.lift_from[j]) for j in (0, 1))
        self.lift_T = max(10, int(np.ceil(dist / p["lift_v"])))

    def _lift_target(self, arm, s):
        """(hand position, wrist yaw change) at path progress s in [0, 1]."""
        p = self.p
        psi = s * self.dpsi
        c, sn = np.cos(psi), np.sin(psi)
        o = self.lift_from[arm][:2] - self.c0
        xy = self.c0 + s * self.dc + np.array([c * o[0] - sn * o[1], sn * o[0] + c * o[1]])
        z = self.lift_from[arm][2] + p["lift_dz"] * smoothstep(s, p["rise_on"], 1.0)
        return np.array([xy[0], xy[1], z]), psi

    def __call__(self, env, t: int) -> np.ndarray:
        p = self.p
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        if self.offset is None:
            self.offset = (cur[:, :3] - cmd[:, :3]).mean(0)
            self._setup(env)
        cmd[:, :3] += self.offset
        tray = env.breakfast_tray
        p0, rt = pose(tray.body)
        targets, ff, rts, grips, nxts = {}, {}, {}, {}, {}
        for arm in (0, 1):
            g = p0 + rt.apply(self.rim_local[arm])  # live grasp point
            ph, grip, nxt, yaw = self.phase[arm], 0.0, None, 0.0
            self.k[arm] += 1
            k = self.k[arm]
            target, ff[arm] = cmd[arm, :3], np.zeros(3)
            if ph == "pre":
                target = g + [0, 0, p["pre_dz"]]
                if np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] or k > 80:
                    nxt = "descend"
            elif ph == "descend":
                target = g
                if np.linalg.norm(target - cur[arm, :3]) < p["grasp_tol"] or k > 30:
                    nxt = "close"
            elif ph == "close":
                target, grip = g, 1.0
                if holding(env, tray, arm) or k > p["close_steps"]:
                    self.events.add(f"held{arm}")
                    nxt = "wait"
            elif ph == "wait":
                target, grip = g, 1.0
                if {"held0", "held1"} <= self.events:
                    if not hasattr(self, "lift_from"):
                        self._lift_plan(cur, p0)
                    nxt = "lift"
            elif ph == "lift":  # both hands on one minimum-jerk profile along the tray's path (feed-forward + P)
                grip = 1.0
                target, yaw = self._lift_target(arm, _minjerk(k / self.lift_T))
                ff[arm] = self._lift_target(arm, _minjerk((k + 1) / self.lift_T))[0] - target
            targets[arm], rts[arm] = np.asarray(target, float), down(self.finger[arm] + yaw)
            grips[arm], nxts[arm] = grip, nxt
        # the two hands leave each approach waypoint together (one bimanual reach, not two separate ones)
        for ph in ("pre", "descend"):
            if self.phase[0] == self.phase[1] == ph and not (nxts[0] and nxts[1]):
                nxts[0] = nxts[1] = None
        a = np.zeros(14, np.float32)
        for arm, (dp, dr) in enumerate(self._servo(cmd, targets, ff, rts)):
            a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
            a[12 + arm] = grips[arm]
            if nxts[arm]:
                self.phase[arm], self.k[arm] = nxts[arm], 0
        return a
