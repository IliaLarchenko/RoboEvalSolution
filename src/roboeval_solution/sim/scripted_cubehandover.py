"""Scripted CubeHandover policy with privileged rod pose (scripts/rollout.py --scripted cubehandover).

The hand on the rod's side (picker) grasps it top-down near the end nearer its own side, fingers across the rod
(of the 180-deg-symmetric grasps the one nearest the current wrist yaw), lifts it to a hand-over point between the
arms while turning it toward the line between the hands only as far as needed to bring it within ~45 deg of that
line (the picked end toward the picker); the other hand (receiver) waits at the picker's height beside the hand-over
point (further out while the rod is still on its way or turning, clear of its sweep), takes the free end top-down
from there while the picker offers the rod 2-4 cm toward it, and once it holds, the picker opens and backs off
-> success (receiver holding, picker not, rod off the table: every stage).

The hand-over gates on what the policy observes (the gripper openings, 0.1 steps): the receiver's close is judged
once its fingers stop; closed on nothing (opening < 0.25) it opens, backs up 3 cm along its approach and comes in
again. The picker opens only after the receiver's opening has stayed at the rod's width (unchanged, >= 0.25)
both_hold (3-10) frames, both hands still meanwhile (never apart while both hold), and backs off only once its own
opening reads open (fingers clear of the rod). A rod slipping out of the receiver while the picker still holds it:
the receiver grasps again.

The picker squares its wrist to the rod above it before going down, and while carrying turns the rod by its measured
axis (a rod sitting askew in the grasp otherwise never lines up). The two arms' joint speed and wrist height
differences are scored (bavd, bgvd): the receiver's height follows the picker's commanded height (with its velocity
as lead, the vertical not speed-capped) and both grasp the rod at the same height.

Motion: per arm a velocity command toward the target (scripted_common.servo: gain kp, |v| <= vmax, |dv| <= amax per
step) on the commanded pose (FK of the actuator targets + an FK->world offset measured at t=0), rotation <= max_dr per
step.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboeval_solution.sim.scripted_common import cmd_pose, down, ee_sites, holding, pose, servo, smoothstep

GRASP_DZ = 0.10  # site above the rod centre line at the grasp (demos / policy 0.095-0.104)
# the picker's table grasp is higher (DEFAULTS pick_dz 0.110) and its descent slower (desc_v): at 0.10 the pads'
# lower edge comes within a few mm of the table and can touch it (an env collision)

DEFAULTS = dict(dev_max=0.82, dev_w=0.15, match_min=0.003, match_gain=0.75, stage_dy=0.06, end_off=0.08, hand_x=0.5,
                hand_y=0.0, pick_dz=0.110, hand_dz=0.10, pre_dz=0.07, kp=0.35, vmax=0.0135, amax=0.0028,
                max_dr=0.04, grasp_tol=0.006, rot_tol=4.0, align_gain=0.3, align_lo=0.08, align_hi=0.25, far_lo=0.03,
                far_hi=0.10, back_dy=0.06, wait_floor=0.055, via_tol=0.02, over_tol=0.012, close_steps=12,
                release_steps=4, back_dz=0.08, recv_tilt=22.0, desc_v=0.6, offer=0.03, both_hold=0,
                # the observed openings (0.1 steps, 1 = open): the receiver's close is judged once its opening has
                # stayed the same steady_n frames: >= grasp_lo is the rod between its fingers, below it closed on
                # nothing -> it opens (until open_ok), backs up retry_up along its approach and comes in again (at
                # most max_regrasp times). The picker opens only after the receiver's opening has stayed at the rod
                # width (>= grasp_lo, unchanged) both_hold frames, and backs off once its own reads open_ok
                grasp_lo=0.25, steady_n=1, open_ok=0.85, reopen_max=8, retry_up=0.03, max_regrasp=3)
RANDOMIZE = dict(dev_max=(0.75, 0.9), stage_dy=(0.04, 0.08), end_off=(0.075, 0.085), hand_x=(0.46, 0.54),
                 hand_y=(-0.04, 0.04), hand_dz=(0.08, 0.13), pre_dz=(0.05, 0.09), kp=(0.3, 0.4), vmax=(0.012, 0.015),
                 amax=(0.0025, 0.003), max_dr=(0.036, 0.044), recv_tilt=(19.0, 25.0),
                 match_gain=(0.65, 0.85), desc_v=(0.5, 0.7), offer=(0.02, 0.04),
                 # the policy's arms tend to move apart before the giving hand opens and the receiving hand loses the
                 # rod; here both hands hold still both_hold frames once the receiver holds, then the picker opens
                 # and stays put release_steps frames before it backs off
                 both_hold=(3, 10), release_steps=(4, 8))


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class ScriptedCubeHandover:
    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}
        self.v = np.zeros((2, 3))
        self.phase: dict[int, str] = {}
        self.k = {0: 0, 1: 0}
        self.events: set[str] = set()
        self.offset = None
        self.op_prev = {0: None, 1: None}  # last observed opening per hand and how many frames it has stayed
        self.op_run = {0: 0, 1: 0}
        self.recv_ok = 0  # frames the receiver has held the rod at a steady opening (closed, >= grasp_lo)
        self.attempt = 0  # receiver closes judged empty so far

    @staticmethod
    def _nearest(cands, cur_rot):
        yc = cur_rot.as_matrix()[:, 1]
        phi = np.arctan2(yc[1], yc[0])
        return min(cands, key=lambda c: abs(_wrap(c - phi)))

    def _setup(self, env, cur):
        c, rr = pose(env.cube.body)
        self.picker = 0 if c[1] > 0 else 1  # the hand on the rod's side
        if getattr(self, "force_picker", None) is not None:  # recovery's retry: the evaluator's first holder
            self.picker = self.force_picker
        self.recv = 1 - self.picker
        side = 1.0 if self.picker == 0 else -1.0  # +y for the left hand
        ax = rr.apply([0.0, 1.0, 0.0])
        self.end_sign = 1.0 if side * ax[1] >= 0 else -1.0  # the end nearer the picker's side (local y sign)
        end_dir = np.arctan2(self.end_sign * ax[1], self.end_sign * ax[0])
        # rod yaw change: the picked end toward the picker by the part of its deviation (<= 90 deg) beyond dev_max (a
        # smooth knee: softplus of width dev_w); the receiver turns its wrist to meet the rod where it ends up. Every
        # turn is out and back for the picker's wrist (square to the rod, then turn it toward home) but one way for
        # the receiver's: a turn of the whole deviation would make the picker's joints travel twice the receiver's
        # (bavd)
        dev = _wrap(np.arctan2(side, 0.0) - end_dir)
        w = self.p["dev_w"]
        excess = w * np.logaddexp(0.0, (abs(dev) - self.p["dev_max"]) / w)
        self.turn = float(np.sign(dev) * min(excess, abs(dev)))
        self.pick_dir = end_dir + self.turn  # world yaw of the picked end at the hand-over
        across = np.arctan2(ax[1], ax[0]) + np.pi / 2  # finger axis across the rod
        self.f_pick = self._nearest([across, across + np.pi], R.from_rotvec(cur[self.picker, 3:]))
        self.f_recv = self._nearest([self.pick_dir + np.pi / 2, self.pick_dir - np.pi / 2], R.from_rotvec(cur[self.recv, 3:]))
        self.phase = {self.picker: "pre", self.recv: "wait_high"}
        self.side = side
        self.corr = 0.0  # yaw correction of the picker's wrist from the rod's measured axis

    def __call__(self, env, t: int) -> np.ndarray:
        p = self.p
        cur = ee_sites(env)
        cmd = cmd_pose(env)
        if self.offset is None:
            self.offset = (cur[:, :3] - cmd[:, :3]).mean(0)
            self._setup(env, cur)
        cmd[:, :3] += self.offset
        rod = env.cube
        c, rr = pose(rod.body)
        ax = rr.apply([0.0, 1.0, 0.0])
        a = np.zeros(14, np.float32)
        hand_c = np.array([p["hand_x"], p["hand_y"], 0.0])
        op = np.asarray(env.robot.qpos_grippers, float)  # observed openings [left, right]
        for arm in (0, 1):
            self.op_run[arm] = self.op_run[arm] + 1 if self.op_prev[arm] == op[arm] else 0
            self.op_prev[arm] = op[arm]
        r = self.recv
        steady_rod = self.phase[r] == "hold" and op[r] >= p["grasp_lo"] and self.op_run[r] >= 1
        self.recv_ok = self.recv_ok + 1 if steady_rod else 0
        for arm in (0, 1):
            ph, grip, nxt = self.phase[arm], 0.0, None
            self.k[arm] += 1
            k = self.k[arm]
            target = cmd[arm, :3]
            if arm == self.picker:
                g = c + self.end_sign * p["end_off"] * ax  # live grasp point on the picked end
                rt = down(self.f_pick)
                if ph == "pre":  # above the grasp, the wrist square to the rod before going down (a turn near the
                    # table brushes it; a rod grasped askew sits askew at the hand-over)
                    target = g + [0, 0, p["pick_dz"] + p["pre_dz"]]
                    rerr = np.degrees((rt * R.from_rotvec(cur[arm, 3:]).inv()).magnitude())
                    if (np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] and rerr < p["rot_tol"]) or k > 90:
                        nxt = "descend"
                elif ph == "descend":
                    target = g + [0, 0, p["pick_dz"]]
                    if np.linalg.norm(target - cur[arm, :3]) < p["grasp_tol"] or k > 30:
                        nxt = "close"
                elif ph == "close":
                    target, grip = g + [0, 0, p["pick_dz"]], 1.0
                    if holding(env, rod, arm) or k > p["close_steps"]:
                        self.z_rest = c[2]
                        nxt = "carry"
                elif ph == "carry":  # rod centre to the hand-over point, turned to lie along y
                    grip = 1.0
                    # closed loop on the rod's own yaw: once the wrist has made its planned turn, a rod sitting
                    # askew in the grasp is turned the rest of the way (the at-handover check is on the rod axis)
                    err = _wrap(np.arctan2(ax[1], ax[0]) - self.pick_dir)
                    err = (err + np.pi / 2) % np.pi - np.pi / 2  # the rod's 180 deg symmetry
                    planned = down(self.f_pick + self.turn + self.corr)
                    if np.degrees((planned * R.from_rotvec(cur[arm, 3:]).inv()).magnitude()) < 3.0:
                        self.corr = float(np.clip(self.corr - p["align_gain"] * err, -0.6, 0.6))
                    rt = down(self.f_pick + self.turn + self.corr)
                    hc = hand_c + [0, 0, self.z_rest + p["hand_dz"]]
                    if self.phase[self.recv] in ("descend", "close", "hold", "reopen", "retry_up"):
                        # offer the rod: it moves toward the receiver while the receiver reaches for it (and stays
                        # there while it reaches again), not the receiver alone (bavd)
                        hc = hc + p["offer"] * np.array([np.cos(self.pick_dir + np.pi), np.sin(self.pick_dir + np.pi), 0.0])
                    target = cur[arm, :3] + (hc - c)  # move the rod centre, the site follows
                    mis = abs(_wrap(np.arctan2(ax[1], ax[0]) - self.pick_dir)) % np.pi  # rod axis vs target, mod 180
                    if np.linalg.norm(hc - c) < p["over_tol"] and min(mis, np.pi - mis) < 0.12:
                        self.events.add("at_handover")
                    if "recv_held" in self.events:  # both hold the rod, still; the picker opens once the receiver's
                        # opening has stayed at the rod width both_hold frames (it holds the rod: no drop on release)
                        target = cmd[arm, :3]
                        if self.recv_ok >= max(1, p["both_hold"]):
                            nxt = "release"
                elif ph == "release":  # open in place; back off only once the fingers are open (clear of the rod)
                    rt = down(self.f_pick + self.turn + self.corr)
                    target = cmd[arm, :3]
                    if k >= p["release_steps"] and (op[arm] >= p["open_ok"] or k >= p["release_steps"] + 10):
                        self.back = cur[arm, :3] + [0, 0.06 * self.side, p["back_dz"]]
                        nxt = "back"
                elif ph == "back":
                    rt = down(self.f_pick + self.turn + self.corr)
                    target = self.back
            else:
                rt = down(self.f_recv)
                # the free end: the rod end on the receiver's side
                free = np.array([np.cos(self.pick_dir + np.pi), np.sin(self.pick_dir + np.pi), 0.0])
                end = 1.0 if np.dot(free, ax) >= 0 else -1.0
                g = c + end * p["end_off"] * ax
                # lean the receiving wrist away from the picker about the finger axis (pads stay square on the rod),
                # so its forearm clears the picker's wrist on the way down (recv_tilt 19-25 deg: >= 3 cm apart at the
                # hand-over)
                th = np.radians(p["recv_tilt"])
                cands = [rt * R.from_rotvec([0.0, sg * th, 0.0]) for sg in (1.0, -1.0)]
                rt = min(cands, key=lambda r: float(np.dot(r.apply([0.0, 0.0, 1.0])[:2], free[:2])))
                zr = rt.apply([0.0, 0.0, 1.0])
                gdz = lambda dz: g - dz * zr  # site with the pads at g (dz = rdz), or further back along the approach
                hc_end = hand_c + p["end_off"] * free
                rdz = p["pick_dz"] / max(-zr[2], 0.5)  # the receiver's site as high above the rod as the picker's
                if ph == "wait_high":  # shadow the picker: staging point beside the hand-over, at the picker's height
                    stage = hc_end + p["stage_dy"] * free
                    # at the picker's height (wrist heights are compared every step: bgvd): its commanded height plus
                    # its vertical velocity / kp (the lag of the P servo), the receiver's pads no lower than
                    # wait_floor above the rod's resting height (>= 2 cm above the top of a rod lying under them)
                    zp = cmd[self.picker, 2] + self.v[self.picker][2] / p["kp"]
                    z = max(zp, getattr(self, "z_rest", c[2]) + GRASP_DZ + p["wait_floor"])
                    mis = abs(_wrap(np.arctan2(ax[1], ax[0]) - self.pick_dir)) % np.pi
                    # the further the rod still has to turn or to travel, the further out the receiver waits, clear of
                    # its sweep (a free end swinging into the open fingers jams the turn; one rising under them jams
                    # the carry): smooth blends, no switch
                    w = smoothstep(min(mis, np.pi - mis), p["align_lo"], p["align_hi"])
                    hc = hand_c + [0, 0, getattr(self, "z_rest", c[2]) + p["hand_dz"]]
                    w = 1.0 - (1.0 - w) * (1.0 - smoothstep(np.linalg.norm(hc - c), p["far_lo"], p["far_hi"]))
                    stage = stage + w * p["back_dy"] * free
                    stage = stage - rdz * np.array([zr[0], zr[1], 0.0])  # the pads (not the site) beyond the free end
                    target = np.array([stage[0], stage[1], z])
                    # in from the staging point only: rising late from below it, the fingers meet the rod's underside
                    if "at_handover" in self.events and (np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] or k > 150):
                        nxt = "descend"
                elif ph == "descend":
                    target = gdz(rdz)
                    # or the command has settled near it: a far reach can sag ~7 mm below the commanded pose
                    err = np.linalg.norm(target - cur[arm, :3])
                    settled = err < 2 * p["grasp_tol"] and np.linalg.norm(self.v[arm]) < 2e-4
                    if err < p["grasp_tol"] or settled or k > 30:
                        nxt = "close"
                elif ph == "close":
                    target, grip = gdz(rdz), 1.0
                    # judged on its own observed opening once that has settled: the rod between the fingers or not
                    if (k >= 2 and self.op_run[arm] >= p["steady_n"]) or k > p["close_steps"]:
                        if op[arm] < p["grasp_lo"] and self.attempt < p["max_regrasp"]:
                            self.attempt += 1  # closed on nothing: open, back up and come in again
                            nxt = "reopen"
                        else:
                            self.events.add("recv_held")
                            nxt = "hold"
                elif ph == "reopen":  # open where it is, then back out along the approach
                    target = cmd[arm, :3]
                    if op[arm] >= p["open_ok"] or k >= p["reopen_max"]:
                        nxt = "retry_up"
                elif ph == "retry_up":
                    target = gdz(rdz + p["retry_up"])
                    if np.linalg.norm(target - cur[arm, :3]) < p["via_tol"] or k > 20:
                        nxt = "descend"
                elif ph == "hold":
                    target, grip = cmd[arm, :3], 1.0
                    if op[arm] < p["grasp_lo"] and self.phase[self.picker] == "carry" \
                            and self.attempt < p["max_regrasp"]:
                        # the rod slipped out of its fingers while the picker still holds it: grasp again
                        self.attempt += 1
                        self.events.discard("recv_held")
                        nxt = "reopen"
            vmax_xy = vmax = None
            if arm == self.picker and ph == "descend":  # slower onto the table: at full speed the descent overshoots
                vmax = p["desc_v"] * p["vmax"]  # the grasp height by up to 9 mm (pads 5 mm above the table)
            if arm == self.recv and ph == "wait_high":  # move with the picker: similar joint speeds (bavd); the
                # vertical uncapped, else the receiver's height lags the picker's (bgvd)
                vmax_xy = max(p["match_min"], p["match_gain"] * float(np.linalg.norm(self.v[self.picker])))
            dp, dr = servo(p, self.v, arm, cmd, np.asarray(target, float), rt, vmax=vmax, vmax_xy=vmax_xy)
            a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
            a[12 + arm] = grip
            if nxt:
                self.phase[arm], self.k[arm] = nxt, 0
        return a
