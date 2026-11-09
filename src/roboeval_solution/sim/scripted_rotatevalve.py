"""Scripted RotateValve policy with privileged valve poses / states (scripts/rollout.py --scripted rotatevalve).

Success: both handwheels turned counterclockwise past state 0.1 (27 deg of their 270 deg range); stages: valve 1
grasped, turned, valve 2 grasped, turned. Both hands work in parallel and in step, each on the valve on its side:

  1. reach on one shared clock: each hand moves to `pre_dz` above its wheel (straight line, minimum-jerk over the
     same T for both arms: set by the longer move at `v_avg`), fingers kept at their current yaw (the wheel is
     round), pre-turned clockwise by `pre_turn`. Once both hands are centred over their wheels (within `xy_tol`:
     the arm lags a fast move by up to a few cm, an early descent puts a fingertip on the rim) both come straight
     down to the demo grasp height (site 11.4 cm above the wheel body; minimum-jerk over `t_down`).
  2. close once the site is within `z_tol` of the grasp height and nearly at rest, until both pads hold the wheel
  3. twist counterclockwise about the wheel axis (world z) once BOTH hands hold: the same smooth rate profile on
     both wrists (ramps up over `ramp` steps to max_dr per step), ramping down once the valve state passes
     `target_state` (the episode ends at success: both past 0.1)
  a hand that loses the wheel before its valve is done opens, goes back up and grasps again (servo below), at a
  corrected height: a blocked twist (over the last 8 twist frames the measured wrist yaw turned < 1/3 of the
  commanded, >= 0.2 rad: a pad caught on the valve body, the grasp too low) regrasps 1.5 cm higher than this grasp
  was; a slip (the observed opening < 0.35 while closed, or the wheel no longer held) 1 cm lower; at most 8 mm above
  / 4 mm below the planned grasp height.

Both arms arrive, descend, close and twist together: the two-arm joint speed difference (bavd) and height difference
(bgvd) are scored, and every motion starts and ends with zero velocity and acceleration (joint jerk is scored).
Env details: a static command settles 4-9 mm below its FK pose (gravity), a moving one tracks it; a hand that
closes 3+ mm low catches a fingertip pad on the valve body during the twist and stalls, hence `sag` / `zb`. On a few
seeds the two valve bases spawn interpenetrating and are flung apart on the first physics step, leaving a valve
tipped over or beyond the top-down reach (x > 0.70).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim.scripted_common import cmd_pose, ee_sites, holding, smoothstep

GRASP_DZ = 0.114  # site above the wheel body origin at the grasp (demos and policy 0.114-0.117)

DEFAULTS = dict(pre_dz=0.03, pre_turn=0.0, target_state=0.15, v_avg=0.006, t_min=24, t_down=12, xy_tol=0.005,
                sag=0.007, z_tol=0.002, ki_z=0.5, close_v=0.001, settle_v=0.0003, close_steps=10, wait_max=15,
                max_dr=0.07, ramp=10, grasp_tol=0.006, kp=0.3, vmax=0.010, amax=0.0025, via_tol=0.015, kr=0.5,
                ar_max=0.006, slow_arm=0, twist_slow=1.0,
                # twist check: over the last block_win frames of a twist the wrist (measured) turned less than
                # 1 / block_ratio of what was commanded (at least block_min rad): blocked, the grasp too low (a pad on
                # the valve body) -> open, regrasp raise_fix higher than where it was; the observed opening fell below
                # grasp_lo while closed (the wheel slipped out): -> regrasp lower_fix lower. The new height is at most
                # fix_hi above / fix_lo below the planned grasp height; at most max_regrasp blocked-twist regrasps
                # per hand (a lost wheel is grasped again every time)
                block_win=8, block_ratio=3.0, block_min=0.2, raise_fix=0.015, lower_fix=0.01, fix_hi=0.008,
                fix_lo=0.004, grasp_lo=0.35, max_regrasp=3,
                # a close that reads empty (opening <= empty_open after empty_k frames, the wheel not held): open and
                # grasp again lower_fix lower at once (else it twists empty for a few frames before the slip check:
                # high empty closes push the wheel round without a hold)
                empty_open=0.05, empty_k=4)
RANDOMIZE = dict(pre_dz=(0.027, 0.034), pre_turn=(0.0, 0.08), target_state=(0.13, 0.17), v_avg=(0.0055, 0.0065),
                 t_min=(20, 28), t_down=(10, 14), max_dr=(0.06, 0.08), ramp=(8, 12),
                 # the policy tends to twist both wrists as one phase that ends when either valve is done (the
                 # lagging valve left at 0-20 deg); here one wrist turns at twist_slow of the rate, so its
                 # valve finishes later and that wrist keeps turning on its own after the other valve is done
                 slow_arm=(0, 1), twist_slow=(0.6, 1.0))


def _yaw(rotvec) -> float:
    """World yaw of the finger axis (site y) of a top-down hand."""
    y = R.from_rotvec(rotvec).as_matrix()[:, 1]
    return float(np.arctan2(y[1], y[0]))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _min_jerk(x: float) -> float:
    """0 -> 1 with zero velocity and acceleration at both ends."""
    x = min(1.0, max(0.0, x))
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


class ScriptedRotateValve:
    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.v = np.zeros((2, 3))
        self.w = np.zeros((2, 3))  # commanded wrist rotation rate (rotvec per step), regrasp servo
        self.phase = {0: "reach", 1: "reach"}
        self.k = {0: 0, 1: 0}
        self.offset = None
        self.grasp_rot: dict[int, R] = {}
        self.rate = {0: 0.0, 1: 0.0}  # twist ramp state in [0, 1]: rate = max_dr * smoothstep
        self.zfix = {0: 0.0, 1: 0.0}  # regrasp height correction (m, + = higher)
        self.hist: dict[int, list] = {0: [], 1: []}  # twist frames: (commanded wrist yaw, measured wrist yaw)
        self.regrasps = {0: [], 1: []}  # ("blocked" | "slip", step) per hand

    def _wheel(self, env, i):
        ph = env._mojo.physics
        if not hasattr(self, "_bids"):
            self._bids = []
            for v in env.valves:
                g = v.valve
                gid = ph.bind(g._mjcf if hasattr(g, "_mjcf") else g.mjcf).element_id
                self._bids.append(int(ph.model.geom_bodyid[gid]))
        return np.asarray(ph.data.xpos[self._bids[i]], float).copy()

    def _servo(self, arm, cmd, target, rt):
        """Velocity + acceleration limited step toward (target, rt): the regrasp approach."""
        p = self.p
        v_des = p["kp"] * (target - cmd[arm, :3])
        n = np.linalg.norm(v_des)
        if n > p["vmax"]:
            v_des *= p["vmax"] / n
        dv = v_des - self.v[arm]
        n = np.linalg.norm(dv)
        if n > p["amax"]:
            dv *= p["amax"] / n
        self.v[arm] = self.v[arm] + dv
        w_des = p["kr"] * (rt * R.from_rotvec(cmd[arm, 3:]).inv()).as_rotvec()
        n = np.linalg.norm(w_des)
        if n > p["max_dr"]:
            w_des *= p["max_dr"] / n
        dw = w_des - self.w[arm]
        n = np.linalg.norm(dw)
        if n > p["ar_max"]:
            dw *= p["ar_max"] / n
        self.w[arm] = self.w[arm] + dw
        return self.v[arm].copy(), self.w[arm].copy()

    def _reach(self, arm, w, t):
        """Shared-clock reach target at step t: minimum-jerk move to pre_dz above the grasp, then (from the shared
        descent start t_d) a minimum-jerk descent to the grasp."""
        p = self.p
        grasp = w + [0, 0, GRASP_DZ]
        pre = grasp + [0, 0, p["pre_dz"]]
        s = _min_jerk(t / self.t_move)
        d = 0.0 if self.t_d is None else _min_jerk((t - self.t_d) / p["t_down"])
        pos = self.start[arm] + s * (pre - self.start[arm]) + d * (grasp - pre)
        rot = self.start_rot[arm] * R.from_rotvec(s * self.rot_delta[arm])
        return pos, rot

    def __call__(self, env, t: int) -> np.ndarray:
        p = self.p
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        if self.offset is None:
            # FK (robot base) -> world, measured at rest right after the reset; a held (static) command settles ~4-9 mm
            # lower than that (gravity; a moving one tracks it), so every commanded height is raised by `sag` and the
            # rest is trimmed per arm before closing (zb)
            self.offset = (cur[:, :3] - cmd[:, :3]).mean(0) - [0.0, 0.0, p["sag"]]
            self.zb = {0: 0.0, 1: 0.0}
            ys = [self._wheel(env, i)[1] for i in range(len(env.valves))]
            left = int(np.argmax(ys))
            self.valve = {0: left, 1: 1 - left}
            self.start, self.start_rot, self.rot_delta = {}, {}, {}
            for arm in (0, 1):  # top-down at the current finger yaw, pre-turned clockwise
                m = R.from_rotvec(cur[arm, 3:]).as_matrix()
                y = m[:, 1].copy(); y[2] = 0.0; y /= np.linalg.norm(y)
                z = np.array([0.0, 0.0, -1.0])
                r0 = R.from_matrix(np.stack([np.cross(y, z), y, z], 1))
                self.grasp_rot[arm] = R.from_rotvec([0, 0, -p["pre_turn"]]) * r0
                self.start[arm] = cmd[arm, :3] + self.offset
                self.start_rot[arm] = R.from_rotvec(cmd[arm, 3:])
                self.rot_delta[arm] = (self.start_rot[arm].inv() * self.grasp_rot[arm]).as_rotvec()
            # one move duration for both arms, from the longer move
            dist = max(np.linalg.norm(self._wheel(env, self.valve[a]) + [0, 0, GRASP_DZ + p["pre_dz"]]
                                      - self.start[a]) for a in (0, 1))
            self.t_move = max(float(p["t_min"]), dist / p["v_avg"])
            self.t_d = None
            self.prev = cur[:, :3].copy()
        cmd[:, :3] += self.offset
        a = np.zeros(14, np.float32)
        if self.t_d is None and t + 1 >= self.t_move:
            # both hands come down together once both are centred over their wheels (the arm lags the commanded
            # move by up to a few cm at speed: an early descent puts a fingertip on the rim)
            off = max(np.linalg.norm((self._wheel(env, self.valve[a]) - cur[a, :3])[:2]) for a in (0, 1))
            if off < p["xy_tol"] or t + 1 >= self.t_move + p["wait_max"]:
                self.t_d = t
        out = {}
        held = {}
        op = np.asarray(env.robot.qpos_grippers, float)  # observed openings [left, right] (0.1 steps, 1 = open)
        for arm in (0, 1):
            obj = env.valves[self.valve[arm]]
            held[arm] = holding(env, obj.valve, arm)
        for arm in (0, 1):
            obj = env.valves[self.valve[arm]]
            w = self._wheel(env, self.valve[arm])
            state = float(np.mean(obj.get_state()))
            ph, grip, nxt, servo = self.phase[arm], 0.0, None, False
            self.k[arm] += 1
            k = self.k[arm]
            grasp = w + [0, 0, GRASP_DZ + self.zfix[arm]]
            target, rt = grasp + [0.0, 0.0, self.zb[arm]], self.grasp_rot[arm]
            if ph == "reach":
                target, rt = self._reach(arm, w, t + 1)
                if self.t_d is not None and t + 1 >= self.t_d + p["t_down"]:
                    # descent commanded: the arm catches up (it lags ~3 steps); close once it is within z_tol of the
                    # grasp height and nearly at rest; a hand that comes to rest outside z_tol trims its height
                    err = grasp - cur[arm, :3]
                    speed = float(np.linalg.norm(cur[arm, :3] - self.prev[arm]))
                    if (abs(err[2]) < p["z_tol"] and np.linalg.norm(err) < p["grasp_tol"] and speed < p["close_v"]) \
                            or k > self.t_move + 40:
                        nxt = "close"
                    elif speed < p["settle_v"]:
                        self.zb[arm] = float(np.clip(self.zb[arm] + p["ki_z"] * err[2], -0.01, 0.01))
                target = target + [0.0, 0.0, self.zb[arm]]
            elif ph == "close":
                grip = 1.0
                other = 1 - arm
                both = held[arm] and (held[other] or self.phase[other] in ("twist", "hold"))
                if k >= p["empty_k"] and not held[arm] and op[arm] <= p["empty_open"]:
                    # closed on nothing (too high / beside): open, back up, grasp again lower, never twisting empty
                    z_err = float(cur[arm, 2] - (w[2] + GRASP_DZ))
                    self.zfix[arm] = float(np.clip(z_err - p["lower_fix"], -p["fix_lo"], p["fix_hi"]))
                    self.regrasps[arm].append(("empty", t))
                    self.grasp_rot[arm] = R.from_rotvec(cur[arm, 3:])
                    self.v[arm], self.w[arm] = 0.0, 0.0
                    grip, nxt = 0.0, "reopen"
                elif both or k > p["close_steps"] + p["wait_max"]:
                    nxt = "twist"
            elif ph in ("twist", "hold"):
                grip = 1.0
                self.rate[arm] = min(1.0, max(0.0, self.rate[arm] + (1.0 if state < p["target_state"] else -1.0)
                                              / p["ramp"]))
                # twist about world z from the commanded orientation: a lagging / failed IK step never piles up
                twist = p["max_dr"] * smoothstep(self.rate[arm], 0.0, 1.0)
                twist *= p["twist_slow"] if arm == p["slow_arm"] else 1.0
                rt = R.from_rotvec([0, 0, twist]) * R.from_rotvec(cmd[arm, 3:])
                # commanded vs executed wrist yaw over the last block_win twist frames
                h = self.hist[arm]
                h.append((_yaw(cmd[arm, 3:]), _yaw(cur[arm, 3:])))
                del h[:-p["block_win"] - 1]
                blocked = slip = False
                if state < p["target_state"]:
                    n_block = sum(1 for kind, _ in self.regrasps[arm] if kind == "blocked")
                    if len(h) > p["block_win"] and n_block < p["max_regrasp"]:
                        c_turn = _wrap(h[-1][0] - h[0][0])
                        e_turn = _wrap(h[-1][1] - h[0][1])
                        blocked = c_turn >= p["block_min"] and c_turn >= p["block_ratio"] * max(e_turn, 0.0)
                    slip = k > 3 and (op[arm] < p["grasp_lo"] or not held[arm])
                if ph == "twist" and state >= p["target_state"]:
                    nxt = "hold"
                elif blocked or slip:
                    # blocked (grasped too low: a pad on the valve body) or the wheel slipped out (too high / off):
                    # open, back up, grasp again at the current yaw, higher or lower than this grasp was
                    z_err = float(cur[arm, 2] - (w[2] + GRASP_DZ))  # this grasp vs the planned height (measured)
                    fix = z_err + (p["raise_fix"] if blocked else -p["lower_fix"])
                    self.zfix[arm] = float(np.clip(fix, -p["fix_lo"], p["fix_hi"]))
                    self.regrasps[arm].append(("blocked" if blocked else "slip", t))
                    self.grasp_rot[arm] = R.from_rotvec(cur[arm, 3:])
                    self.rate[arm] = 0.0
                    self.v[arm], self.w[arm] = 0.0, 0.0
                    h.clear()
                    nxt = "reopen"
            elif ph == "reopen":
                servo = True
                if k > 4:
                    nxt = "pre"
            elif ph == "pre":
                servo, target = True, grasp + [0, 0, p["pre_dz"]]
                if np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] or k > 80:
                    nxt = "descend"
            elif ph == "descend":
                servo = True
                if np.linalg.norm(grasp - cur[arm, :3]) < p["grasp_tol"] or k > 30:
                    nxt = "close"
            if servo:
                dp, dr = self._servo(arm, cmd, np.asarray(target, float), rt)
            else:
                dp = np.asarray(target, float) - cmd[arm, :3]
                dr = (rt * R.from_rotvec(cmd[arm, 3:]).inv()).as_rotvec()
            out[arm] = (dp, dr, grip, nxt)
        self.prev = cur[:, :3].copy()
        for arm in (0, 1):
            dp, dr, grip, nxt = out[arm]
            a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
            a[12 + arm] = grip
            if nxt:
                self.phase[arm], self.k[arm] = nxt, 0
        return a
