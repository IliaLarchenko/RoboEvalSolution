"""Scripted LiftPot policy with privileged pot pose (scripts/rollout.py --scripted liftpot).

Success: the pot 10 cm above its start, level (within 20 deg), off the cabinets; stages: left grasp, right grasp,
lifted, lifted level with both hands holding. Both hands in parallel:

  1. each hand above its handle's outer bar (left hand the +y handle), turned to the grasp orientation before it
     goes down: finger axis ~80 deg across the bar (nearly radial), approach tilted ~29 deg about the finger axis
     (wrist toward the robot: the far hand of a turned pot reaches it) and leaned ~19 deg toward the pot centre
     about the bar (palm above the rim). Straight down, fingers open, open_out (1 cm) outside the bar centre: the
     inner finger comes down midway between the rim lip and the bar, >= 4 mm from both (kinematic check)
  2. both hands close together once both are down (pad_bias, along the approach, takes up the arm's sag: the
     measured fingertip pads centred on the bar), centring on the bar while the fingers close; nothing moves
     until each hand holds with a steady opening (above hold_min, changing < hold_dq per frame) for hold_steady
     frames (sync: state-gated, ~0.5 s after the fingers stop)
  3. lift both together on one shared height profile (the pot stays level) to `lift_h` above the grasp, slowly
     (slow_v) until the pot is slow_h up, then at lift_v

The lift is closed loop on the pot's actual height (commanded up to lift_max until it is lift_goal = 12.5 cm up:
the arms sag under the load), holds while the pot tilts more than tilt_hold (8 deg) and levels it by raising the
lower handle's hand; far pots are pulled in to x_goal while rising.

The grasp is nearly radial rather than the demos' diagonal one (finger axis ~60 deg), which puts the inner finger
into the rim lip. Servo: scripted_common.servo."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim.scripted_common import cmd_pose, ee_sites, holding, opening, pad_mid, pose, servo, smoothstep

PAD = 0.1034  # site -> fingertip pad centre along the approach axis

HALF_OPEN = 0.0425  # open fingertips this far from the site along the finger axis

DEFAULTS = dict(pull=0.06, lean_deg=19.0, tilt_deg=29.0, inner_x=0.0, y_left=0.235, y_right=-0.246, grasp_z=0.143, ang_left=-80.0, ang_right=80.0,
                pre_dz=0.06, lift_h=0.14, lift_v=0.010, close_steps=12, kp=0.35, vmax=0.016,
                amax=0.004, max_dr=0.08, grasp_tol=0.006, via_tol=0.015, lift_goal=0.125, lift_max=0.30,
                x_goal=0.62, level_gain=0.5, level_max=0.06, tilt_hold=8.0, drag_v=0.008, drag_lift=0.02,
                finger_ref=60.0, open_out=0.0105, rot_tol=5.0, lag=0,
                pad_gain=0.5, pad_max=0.012, pad_conv=0.003, hold_steady=8, hold_dq=0.01,
                hold_min=0.1, sync_max=40, slow_h=0.02, slow_ramp=0.01, slow_v=0.003,
                # per-hand regrasp (one hand holding the pot, the other at a missed grasp: the holding hand stays
                # still and the other grasps properly before anything lifts): a closed hand
                # reading an empty opening (< hold_min, not holding) for empty_n frames opens where it is (until
                # open_ok, at most reopen_max frames), rises over its handle as the pot now lies and grasps it again
                # (pre, descend, close), at most max_regrasp times; the other hand holds still meanwhile and nothing
                # lifts before both holds are steady again
                empty_n=3, open_ok=0.85, reopen_max=8, max_regrasp=2)
# A bar is easily lost when the lift starts right after the close or with the bar off the pads' centre: the lift
# waits for each hand's opening to be steady for hold_steady frames (state-gated, no draw) and rises slowly (slow_v)
# for the pot's first slow_h. The closed opening is the bar's chord along the finger axis, widest (~0.28) with the
# pads centred on the bar; the arm sags a few mm past the centre, and pad_bias centres the measured pads while the
# hand comes down / waits (no extra wait).
# The grasp geometry ranges keep >= 4 mm between the open inner finger and both the rim lip and the bar (kinematic
# check over the corners); wider lean / angle / height ranges eat into that.
RANDOMIZE = dict(pull=(0.05, 0.08), lean_deg=(18.0, 20.0), tilt_deg=(27.0, 31.0), inner_x=(-0.004, 0.004),
                 grasp_z=(0.142, 0.144), ang_left=(-82.0, -78.0), ang_right=(78.0, 82.0), pre_dz=(0.045, 0.075),
                 lift_h=(0.13, 0.15), lift_v=(0.009, 0.012), kp=(0.3, 0.4), vmax=(0.014, 0.018), amax=(0.0035, 0.0045),
                 drag_lift=(0.015, 0.03),
                 # the policy tends to close both hands on one clock, ready or not, the higher hand slipping off its
                 # handle; here one hand starts up to 15 frames late (+ = the right), so the data shows a hand
                 # waiting open at its handle for the other before both close
                 lag=(-15, 15))


class ScriptedLiftPot:
    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.v = np.zeros((2, 3))
        self.phase = {0: "pre", 1: "pre"}
        self.k = {0: 0, 1: 0}
        self.offset = None
        self.lift = 0.0  # shared commanded lift above the grasp
        self.ready = {0: False, 1: False}  # at the grasp point, fingers open around the bar
        self.level = np.zeros(2)  # per-hand height correction keeping the pot level
        self.steady = {0: 0, 1: 0}  # frames each closed hand has held with a steady opening
        self.pad_bias = np.zeros(2)  # per-hand correction along the approach centring the measured pads on the bar
        self.o_prev = {0: None, 1: None}
        self.attempt = {0: 0, 1: 0}  # regrasps so far, per hand
        self.empty = {0: 0, 1: 0}  # frames a closed hand has read empty

    def __call__(self, env, t: int) -> np.ndarray:
        p = self.p
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        pot = env.kitchenpot
        pp, pr = pose(pot.body)
        if self.offset is None:
            self.offset = (cur[:, :3] - cmd[:, :3]).mean(0)
            self.z0 = float(pp[2])
            # far pots are pulled in further while rising: the arms lift a heavy pot better nearer the body
            self.pull = max(p["pull"], float(pp[0]) - p["x_goal"])
            yaw = float(pr.as_euler("xyz")[2])
            self.rot = {}
            for arm, ang in ((0, p["ang_left"]), (1, p["ang_right"])):
                a = yaw + np.radians(ang)
                yc = R.from_rotvec(cur[arm, 3:]).as_matrix()[:, 1]
                # 180 deg finger symmetry: the representation nearest a fixed reference finger_ref off the home finger
                # yaw (mirrored per hand), which keeps every episode on the same side (>= 40 deg from the switch over
                # the pot yaws and finger angles; the representation nearest the wrist would switch at 90 deg)
                ref = np.arctan2(yc[1], yc[0]) + np.radians(p["finger_ref"] * (-1.0 if arm == 0 else 1.0))
                if np.cos(a - ref) < 0:
                    a += np.pi
                y = np.array([np.cos(a), np.sin(a), 0.0])
                z = np.array([0.0, 0.0, -1.0])
                r0 = R.from_matrix(np.stack([np.cross(y, z), y, z], 1))
                # tilt the approach about the finger axis, leaning away from the robot (+x): the wrist sits nearer
                # (~29 deg: below ~20 the far hand of a far turned pot cannot turn its wrist to the grasp)
                b = np.radians(p["tilt_deg"])
                cands = [R.from_rotvec(sg * b * y) * r0 for sg in (1.0, -1.0)]
                r1 = max(cands, key=lambda r: r.as_matrix()[0, 2])
                # lean the approach toward the pot centre about the handle bar (pot x) axis: wrist and palm move
                # outward, the palm (20 cm along the finger axis, reaching over the pot) clear of the rim
                inward = pr.apply([0.0, -1.0 if arm == 0 else 1.0, 0.0])
                bar = pr.apply([1.0, 0.0, 0.0])
                c = np.radians(p["lean_deg"])
                cands = [R.from_rotvec(sg * c * bar) * r1 for sg in (1.0, -1.0)]
                self.rot[arm] = max(cands, key=lambda r: float(np.dot(r.as_matrix()[:, 2], inward)))
            # pad targets in the pot frame (the site sits PAD back along the approach axis)
            # straddle the bar: fingertips 4.25 cm*sin(angle) either side of its centre line, the pads' midpoint at
            # the bar's mid height; the inner fingertip lands at x = inner_x, inside the handle loop clear of its arms
            self.pad_local, self.pad_open = {}, {}
            for arm, ang, yb in ((0, p["ang_left"], p["y_left"]), (1, p["ang_right"], p["y_right"])):
                xc = p["inner_x"] - HALF_OPEN * abs(np.cos(np.radians(ang)))
                self.pad_local[arm] = np.array([xc, yb, p["grasp_z"] - PAD])
                # the open hand comes down open_out further out (the inner finger midway between the rim lip and the
                # bar) and centres on the bar while the fingers close
                self.pad_open[arm] = self.pad_local[arm] + [0.0, p["open_out"] * (1.0 if arm == 0 else -1.0), 0.0]
        cmd[:, :3] += self.offset
        a = np.zeros(14, np.float32)
        held = {arm: holding(env, pot, arm) for arm in (0, 1)}
        for arm in (0, 1):  # a closed hand's hold is steady: holding, the fingers open on the bar and no longer moving
            if self.phase[arm] in ("close", "sync"):
                o = opening(env, arm)
                still = self.o_prev[arm] is not None and abs(o - self.o_prev[arm]) < p["hold_dq"]
                self.steady[arm] = self.steady[arm] + 1 if (held[arm] and o > p["hold_min"] and still) else 0
                self.o_prev[arm] = o
                self.empty[arm] = self.empty[arm] + 1 if (self.phase[arm] == "sync" and not held[arm]
                                                          and o < p["hold_min"]) else 0
        # the lift rate: slow_v until the pot is slow_h up (the bar settles in the closed fingers under the load),
        # then smoothly up to lift_v
        rate = p["slow_v"] + (p["lift_v"] - p["slow_v"]) * smoothstep(float(pp[2] - self.z0), p["slow_h"],
                                                                      p["slow_h"] + p["slow_ramp"])
        # draw the pot in barely lifted by however far it sits beyond x_goal (0 for near pots: one rule, no switch
        # at a distance), then lift while pulling the usual `pull` more
        drag_total = max(0.0, self.pull - p["pull"])
        if all(self.phase[j] == "lift" for j in (0, 1)) and getattr(self, "drag", 0.0) < drag_total - 1e-6:
            self.drag = min(getattr(self, "drag", 0.0) + p["drag_v"], drag_total)
            self.lift = min(self.lift + rate, p["drag_lift"])
        elif all(self.phase[j] == "lift" for j in (0, 1)):
            # closed loop on the pot's actual height: keep commanding higher (the arms sag under the load) until
            # it is lift_goal up, a margin over the 10 cm success threshold
            tilt = float(np.degrees(np.arccos(np.clip(pr.apply([0.0, 0.0, 1.0])[2], -1.0, 1.0))))
            if pp[2] - self.z0 < p["lift_goal"] and tilt < p["tilt_hold"]:  # tilted: hold the height, level first
                self.lift = min(self.lift + rate, p["lift_max"])
            # level: raise the lower handle's hand, lower the higher one's
            hz = [float((pp + pr.apply([0.0, 0.24 if arm == 0 else -0.24, 0.0]))[2]) for arm in (0, 1)]
            d = hz[0] - hz[1]
            self.level = np.clip(self.level + p["level_gain"] * np.array([-d, d]) / 2, -p["level_max"], p["level_max"])
        for arm in (0, 1):
            if t < (p["lag"] if arm == 1 else -p["lag"]):  # the late hand holds still until its start
                self.v[arm] = 0.0
                continue
            ph, grip, nxt = self.phase[arm], 0.0, None
            self.k[arm] += 1
            k = self.k[arm]
            if ph in ("pre", "descend", "close", "sync"):
                pad = self.pad_open[arm] if ph in ("pre", "descend") else self.pad_local[arm]
                # follows the pot; pad_bias along the approach centres the measured pads on the bar (the arm sags
                # below its command)
                ax = self.rot[arm].as_matrix()[:, 2]
                g0 = pp + pr.apply(pad) - PAD * ax  # the site where the pads are where they should be
                g = g0 - self.pad_bias[arm] * ax
                self.g_world = getattr(self, "g_world", {})
                self.g_world[arm] = g
            g = self.g_world[arm]
            target = g + [0, 0, p["pre_dz"]]
            if ph == "pre":
                # down only once above the bar AND turned to the grasp orientation (near pots reach the via point
                # before the wrist has turned, and a finger would come down on the bar)
                rerr = np.degrees((self.rot[arm] * R.from_rotvec(cur[arm, 3:]).inv()).magnitude())
                if (np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] and rerr < p["rot_tol"]) or k > 80:
                    nxt = "descend"
            elif ph == "descend":
                target = g
                if p["pad_gain"] > 0 and np.linalg.norm(target - cmd[arm, :3]) < p["pad_conv"]:
                    # the command has arrived: what is left is the arm's sag. The open pads' midpoint, along the
                    # approach, onto the bar centre: the closed fingers then hold the bar's widest chord (no wait for
                    # it: the correction goes on while the hand waits for the other and while the fingers close)
                    err = float((pad_mid(env, arm) - pp - pr.apply(self.pad_open[arm])) @ ax)
                    self.pad_bias[arm] = float(np.clip(self.pad_bias[arm] + p["pad_gain"] * err, -p["pad_max"],
                                                       p["pad_max"]))
                # latched: a hand that got there waits there (no flicker of the both-ready decision)
                self.ready[arm] |= bool(np.linalg.norm(g0 - cur[arm, :3]) < p["grasp_tol"] or k > 30)
                # close both hands together (open fingers straddling both bars): one hand closing alone kicks the
                # free pot along its bar (the far hand then chases it out of reach)
                if self.ready[arm] and self.ready[1 - arm]:
                    nxt = "close"
            elif ph == "close":
                target, grip = g, 1.0
                if held[arm] or k > p["close_steps"]:
                    nxt = "sync"
            elif ph == "sync":  # both hands hold still until both holds are steady for hold_steady frames
                target, grip = g, 1.0
                if self.phase[1 - arm] not in ("close", "sync", "lift"):  # the other hand grasps again: this one holds
                    # still where it is (following the pot it holds would walk it up) and no timeout runs
                    self.k[arm] = k = 0
                    self.hold_at = getattr(self, "hold_at", {})
                    target = self.hold_at.setdefault(arm, cmd[arm, :3].copy())
                elif getattr(self, "hold_at", None):
                    self.hold_at.pop(arm, None)
                if self.empty[arm] >= p["empty_n"] and self.attempt[arm] < p["max_regrasp"]:
                    # closed on nothing (the fingers shut, no hold): open here, then over the handle and again
                    self.attempt[arm] += 1
                    self.ready[arm], self.steady[arm], self.empty[arm], self.o_prev[arm] = False, 0, 0, None
                    nxt, grip = "reopen", 0.0
                elif all(self.phase[j] in ("sync", "lift") for j in (0, 1)) and \
                        all(self.steady[j] >= p["hold_steady"] for j in (0, 1)):
                    nxt = "lift"
                elif k > p["sync_max"]:
                    nxt = "lift"
            elif ph == "reopen":  # an empty hand opens where it is, then rises over its handle (pre)
                target = cmd[arm, :3]
                if opening(env, arm) >= p["open_ok"] or k >= p["reopen_max"]:
                    nxt = "pre"
            elif ph == "lift":
                f = min(1.0, self.lift / p["lift_h"])  # pull toward the robot as it rises: lifting nearer the body
                pull = getattr(self, "drag", 0.0) + min(self.pull, p["pull"]) * f
                target, grip = g + [-pull, 0, self.lift + self.level[arm]], 1.0
            dp, dr = servo(p, self.v, arm, cmd, np.asarray(target, float), self.rot[arm])
            a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
            a[12 + arm] = grip
            if nxt:
                self.phase[arm], self.k[arm] = nxt, 0
        return a
