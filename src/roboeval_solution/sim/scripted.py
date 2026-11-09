"""Registry of the privileged scripted policies (rollout.py --scripted <name>): name -> task family + controller."""
from __future__ import annotations

import importlib

# name: (task family, module, controller class); each module has DEFAULTS and RANDOMIZE (per-episode uniform ranges)
POLICIES = {
    "packbox": ("PackBox", "roboeval_solution.sim.scripted_packbox", "ScriptedPackBox"),
    "stack": ("StackTwoBlocks", "roboeval_solution.sim.scripted_stack", "ScriptedStack"),
    "rotatevalve": ("RotateValve", "roboeval_solution.sim.scripted_rotatevalve", "ScriptedRotateValve"),
    "liftpot": ("LiftPot", "roboeval_solution.sim.scripted_liftpot", "ScriptedLiftPot"),
    "lifttray": ("LiftTray", "roboeval_solution.sim.scripted_lifttray", "ScriptedLiftTray"),
    "cubehandover": ("CubeHandover", "roboeval_solution.sim.scripted_cubehandover", "ScriptedCubeHandover"),
    "pickbook": ("PickSingleBookFromTable", "roboeval_solution.sim.scripted_pickbook", "ScriptedPickBook"),
    "shelfbook": ("StackSingleBookShelf", "roboeval_solution.sim.scripted_shelfbook", "ScriptedShelfBook"),
}


def family(name: str) -> str:
    if name not in POLICIES:
        raise ValueError(f"unknown scripted policy {name!r}; known: {sorted(POLICIES)}")
    return POLICIES[name][0]


def for_family(fam: str) -> str:
    """The scripted policy that acts in task family `fam`."""
    names = [n for n, (f, _, _) in POLICIES.items() if f == fam]
    if not names:
        raise ValueError(f"no scripted policy for {fam!r}")
    return names[0]


def make(name: str, seed: int, randomize: bool = True, retry: bool = False):
    """(controller, params): a fresh controller for one episode, params drawn from `seed` when randomize. retry: wrapped
    in Retry (recovery's "retry" stage: taking over a failed rollout's state), its own params under "retry"."""
    family(name)
    _, module, cls = POLICIES[name]
    from roboeval_solution.sim.scripted_common import draw_params

    mod = importlib.import_module(module)
    params = draw_params(mod.RANDOMIZE, seed) if randomize else {}
    ctl = getattr(mod, cls)(**params)
    if retry:
        rp = draw_params(RETRY_RANDOMIZE, seed + 1)
        return Retry(ctl, family(name), **rp), {**params, "retry": rp}
    return ctl, params


# Retry: open both hands, rise (and back off toward the robot) clear of the props, hold still while they settle;
# then the family's controller, fresh, plans from the state it finds on its own clock (it measures its FK -> world
# offset at its first call: the arms are still then)
RETRY_DEFAULTS = dict(open_steps=6, rise=0.08, back=0.015, rise_v=0.012, rise_steps=25, settle=8, kp=0.35,
                      amax=0.003, max_dr=0.05)
RETRY_RANDOMIZE = dict(open_steps=(4, 8), rise=(0.06, 0.10), back=(0.0, 0.03), rise_v=(0.010, 0.014), settle=(6, 12))


class Retry:
    """Recovery's "retry" stage (sim/recovery.py): a failed rollout's state, the hands possibly closed on or around a
    slipped, dropped or misplaced object, then a fresh attempt by the family's scripted policy. CubeHandover: the
    evaluator has fixed the hand that held the rod first (it must end up not holding), so that hand picks again."""

    def __init__(self, inner, fam: str, **params):
        import numpy as np

        self.inner, self.fam = inner, fam
        self.p = {**RETRY_DEFAULTS, **params}
        self.v = np.zeros((2, 3))
        self.k, self.phase, self.t0 = 0, "open", None

    def __call__(self, env, t: int):
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        from roboeval_solution.sim.scripted_common import cmd_pose, servo

        p = self.p
        if self.phase == "inner":
            return self.inner(env, t - self.t0)
        a = np.zeros(14, np.float32)  # grippers open, arms still
        self.k += 1
        if self.phase == "open" and self.k >= p["open_steps"]:
            self.phase, self.k = "rise", 0
            self.start = cmd_pose(env)
        elif self.phase == "rise":
            cmd = cmd_pose(env)
            done = True
            for arm in (0, 1):
                target = self.start[arm, :3] + [-p["back"], 0.0, p["rise"]]
                rt = R.from_rotvec(self.start[arm, 3:])  # the wrist as it was
                dp, dr = servo({**p, "vmax": p["rise_v"]}, self.v, arm, cmd, target, rt, stop_cap=True)
                a[6 * arm: 6 * arm + 3], a[6 * arm + 3: 6 * arm + 6] = dp, dr
                done &= bool(np.linalg.norm(target - cmd[arm, :3]) < 0.01)
            if done or self.k >= p["rise_steps"]:
                self.phase, self.k = "settle", 0
                self.v[:] = 0.0
        elif self.phase == "settle" and self.k >= p["settle"]:
            self.phase, self.t0 = "inner", t + 1
            if self.fam == "CubeHandover" and getattr(env, "initial_gripper", None) is not None:
                from roboeval.const import HandSide

                self.inner.force_picker = 0 if env.initial_gripper == HandSide.LEFT else 1
        return a
