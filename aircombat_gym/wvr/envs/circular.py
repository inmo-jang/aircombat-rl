"""Assignment 01 -- kill a loitering target.

    import gymnasium as gym, aircombat_gym.wvr.envs
    env = gym.make("AirCombat/Circular-v0")

Red holds 20,000 ft, turns at a fixed average rate drawn from 1-5 deg/s, flies
at 325 kt, and never shoots.  Blue spawns on a random bearing 1.5-4 km out,
pointed roughly at it, at the same altitude.  Blue wins by holding the firing
envelope; nothing else ends the episode except the 120 s clock.

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
              constant 1.0/s after a 1.0 s lock means exactly one thing: hold it
              for two seconds.
"""
from __future__ import annotations

import math

from ..baselines import Circler
from .base import HEADING, SPEED, Initial, TaskEnv

ALT_FT = 20_000.0                       # both aircraft, fixed
RED_SPEED_KT = 325.0
TURN_LO, TURN_HI = 1.0, 5.0             # deg/s, average; sign random
RANGE_LO_M, RANGE_HI_M = 1500.0, 4000.0
HEADING_CONE_DEG = 90.0                 # blue's heading, either side of the LOS
BLUE_SPEED_LO_KT, BLUE_SPEED_HI_KT = 300.0, 600.0


class CircularTargetEnv(TaskEnv):
    """Assignment 01.  `Discrete(9)` = heading 3 x speed 3, or `Box(2)`."""

    channels = (HEADING, SPEED)         # the vertical is closed
    t_max = 120.0
    track_lock = 1.0                    # seconds to earn the shot
    flat_damage = 1.0                   # then 1.0 s of firing to kill
    wez_cone_deg = 30.0
    lose_on_exit = False                # no arena; leaving is not a loss

    def sample(self, rng):
        r = rng.uniform(RANGE_LO_M, RANGE_HI_M)
        bearing = rng.uniform(-math.pi, math.pi)
        bx, by = r * math.sin(bearing), r * math.cos(bearing)
        toward_red = math.atan2(-bx, -by)
        psi = toward_red + rng.uniform(-math.radians(HEADING_CONE_DEG),
                                       math.radians(HEADING_CONE_DEG))
        ic = Initial(
            "circular",
            ((bx, by), (0.0, 0.0)),                  # blue is red-relative
            (math.degrees(psi), 0.0),
            (rng.uniform(BLUE_SPEED_LO_KT, BLUE_SPEED_HI_KT), RED_SPEED_KT),
            (ALT_FT, ALT_FT),
        )
        rate = rng.uniform(TURN_LO, TURN_HI)
        if rng.random() < 0.5:
            rate = -rate
        return ic, Circler(rate)
