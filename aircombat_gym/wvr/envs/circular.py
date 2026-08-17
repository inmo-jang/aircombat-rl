"""`AirCombat/Circular-v0` -- kill a loitering target.

    import gymnasium as gym, aircombat_gym.wvr.envs
    env = gym.make("AirCombat/Circular-v0")

The learner is **red**, as in every task; the target is **blue**.

Blue holds 20,000 ft, turns at a fixed average rate drawn from 1-5 deg/s and
flies at 325 kt.  Red spawns on a random bearing 1.5-4 km out, pointed roughly
at it, at the same altitude.  Red wins by holding the firing envelope; the only
other end is the 120 s clock.

Blue never *aims*, but it keeps its gun, on purpose.  It cannot manoeuvre for a
shot -- its heading is on rails -- so it is no threat to a policy that stays out
of its windscreen, and every death is one red flew into.  That is a lesson worth
charging for at rung 1, and it costs almost nothing: 0.16 s of solution per match
against `ace`, 0.32 s against a random policy, one kill in twelve random matches.
Part of the 0.26 random baseline is those deaths rather than timeouts.

`foe_armed = False` would disarm it.  Nothing on the ladder uses that now:
`Evader` did, on the argument that a target which runs gets its nose onto
things in a way one on rails cannot, and that is still true -- but a uniform
rule across every rung was judged worth more than the guarantee, so `Evader`
is armed too and its baselines were re-measured against the armed target.

Every number below was measured before the task was fixed; the workings are in
`project_01_circular.md`.  The three that were not free choices:

  altitude    fixed, and the altitude channel is removed from the action.  The
              vertical *was* the difficulty: `ace` scores 0.30 with a +-5,000 ft
              spread and 0.97 co-altitude, and the knee sits exactly where the
              geometry puts it -- 1,000 ft at 800 m is 21 deg of elevation
              against what was then a 15 deg cone.
  cone 30     widened from 15, and not because the aircraft wobbles: with the
              altitude channel shut the drift is 95 ft at p95, which is 2.1 deg
              at 800 m.  It is widened for *exploration*.  A random policy sits
              in the envelope 0.48 % of the time at 15 deg and 0.79 % at 30, and
              that 1.6x is what a beginner's reward function needs to have
              anything to reinforce.  Stopping at 30 rather than 35 keeps the
              hand-written reference off the ceiling, so the top of the class is
              still separable.
  flat damage the honest weapon scales damage by aim error, range and aspect at
              once, so a student cannot work out how long to hold the shot.  A
              constant 0.33/s after a 1.0 s lock means exactly one thing: hold
              it for three seconds after earning it, four in all.  Every task on
              the ladder uses the same number.
"""
from __future__ import annotations

import math

from ..baselines import Circler
from .base import HEADING, SPEED, Initial, TaskEnv

ALT_FT = 20_000.0                       # both aircraft, fixed
FOE_SPEED_KT = 325.0
TURN_LO, TURN_HI = 1.0, 5.0             # deg/s, average; sign random
RANGE_LO_M, RANGE_HI_M = 1500.0, 4000.0
HEADING_CONE_DEG = 90.0                 # red's heading, either side of the LOS
OWN_SPEED_LO_KT, OWN_SPEED_HI_KT = 300.0, 600.0


class CircularTargetEnv(TaskEnv):
    """`Discrete(9)` = heading 3 x speed 3, or `Box(2)`."""

    channels = (HEADING, SPEED)         # the vertical is closed
    t_max = 120.0
    track_lock = 1.0                    # seconds to earn the shot
    flat_damage = 0.33                  # then 3.0 s of firing to kill
    wez_cone_deg = 30.0
    foe_armed = True                    # stated, not inherited -- see above

    def sample(self, rng):
        r = rng.uniform(RANGE_LO_M, RANGE_HI_M)
        bearing = rng.uniform(-math.pi, math.pi)
        bx, by = r * math.sin(bearing), r * math.cos(bearing)
        toward_foe = math.atan2(-bx, -by)
        psi = toward_foe + rng.uniform(-math.radians(HEADING_CONE_DEG),
                                       math.radians(HEADING_CONE_DEG))
        ic = Initial(
            "circular",
            ((bx, by), (0.0, 0.0)),                  # red is placed relative to blue
            (math.degrees(psi), 0.0),
            (rng.uniform(OWN_SPEED_LO_KT, OWN_SPEED_HI_KT), FOE_SPEED_KT),
            (ALT_FT, ALT_FT),
        )
        rate = rng.uniform(TURN_LO, TURN_HI)
        if rng.random() < 0.5:
            rate = -rate
        return ic, Circler(rate)
