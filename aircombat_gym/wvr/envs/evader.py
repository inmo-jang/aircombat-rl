"""`AirCombat/Evader-v0` -- kill a target that breaks when you close.

    import gymnasium as gym, aircombat_gym.wvr.envs
    env = gym.make("AirCombat/Evader-v0")

Everything is `Circular` except the target.  Same altitude, same closed
vertical, same `Discrete(9)`, same cone and lock, same clock.  Blue now cruises
until red is inside 2.5 km and forward of its 3/9 line, then breaks hard for a
random 2-4 s in a random direction.  It is armed, like every target on the
ladder.

**One thing changes and it changes the lesson.**  `Circular`'s target is a
fixed curve: its position two seconds from now is a function of the clock, so a
policy can score without ever reading the opponent's state -- and a mediocre
observation still gets most of the way there.  A break is a function of *where
red is*, so the same trick is worth nothing.  The intended difficulty is not the
turn; it is that the answer stopped being open-loop.

The break parameters are drawn from the env's generator, not from `random`, so a
seed still names an engagement (`main_dqn.py` seed bands).

**The clock is 240 s, not 120.**  Tuning the target could not make this task
tell tactics apart -- slowing it to 380-425 kt makes `ace` and `pursuit` both
score 1.00 across sixteen parameter cells, tightening the gun to a 15 deg cone
buys +0.02, and decelerating into the break buys nothing.  The clock did it:
`ace` 0.87 against `pursuit` 0.65, where the 120 s version was 0.77 against 0.67.
More time helps only the pilot who can convert it.

Two numbers are worth knowing before designing a reward:

  the break is survivable    the F-16 sustains ~12 deg/s at 20,000 ft and the
                             break commands 30 deg of heading; red cannot follow
                             it turn for turn from close in, which is the whole
                             point of `_trackable` in `Circular`'s worked reward.
  breaking costs blue        a 3 s break at 30 deg bleeds it toward red's guns
                             on the way out, so waiting is a real strategy and a
                             reward that pays only for closing will miss it.
"""
from __future__ import annotations

import math

from ..baselines import Evader
from .base import HEADING, SPEED, Initial, TaskEnv

ALT_FT = 20_000.0                       # both aircraft, fixed
FOE_SPEED_KT = 325.0
RANGE_LO_M, RANGE_HI_M = 1500.0, 4000.0
HEADING_CONE_DEG = 90.0                 # red's heading, either side of the LOS
OWN_SPEED_LO_KT, OWN_SPEED_HI_KT = 300.0, 600.0
FOE_HEADING_SPREAD_DEG = 180.0          # blue's initial heading is free


class EvadingTargetEnv(TaskEnv):
    """`Discrete(9)` = heading 3 x speed 3, or `Box(2)`."""

    channels = (HEADING, SPEED)         # the vertical is closed
    # 240 s, twice `Circular`'s.  The clock was the binding constraint and it
    # bound *asymmetrically*: lost matches ended with the range still closing at
    # 19 m/s from 2.8 km, and going to 240 s lifted `ace` from 0.77 to 0.85 while
    # leaving `pursuit` at 0.67.  360 s adds nothing -- the longest kill `ace`
    # needs is 160 s -- so this is the knee, not a round number.
    t_max = 240.0
    track_lock = 1.0
    flat_damage = 0.33
    wez_cone_deg = 30.0
    # Armed, like every other rung.  It was disarmed until 2026-08-17 on a real
    # argument -- a target that runs sweeps its nose across the chaser in a way
    # `Circular`'s rail-bound circler cannot, so the free shots are worth
    # more here -- and the argument was traded for uniformity: once policies
    # can be put in either seat, "the weapon rule is the same everywhere" is
    # the property that makes a ladder out of separate tasks.  The baselines
    # below were re-measured after arming it.
    foe_armed = True

    def sample(self, rng):
        r = rng.uniform(RANGE_LO_M, RANGE_HI_M)
        bearing = rng.uniform(-math.pi, math.pi)
        bx, by = r * math.sin(bearing), r * math.cos(bearing)
        toward_foe = math.atan2(-bx, -by)
        psi = toward_foe + rng.uniform(-math.radians(HEADING_CONE_DEG),
                                       math.radians(HEADING_CONE_DEG))
        # Blue's heading is free rather than fixed at 0.  With the target
        # loitering, `Circular` could leave it: the circle looks the same
        # from anywhere on it.  A cruising target that runs in one direction
        # cannot, and pinning it would let a policy learn a compass bearing.
        foe_psi = rng.uniform(-math.radians(FOE_HEADING_SPREAD_DEG),
                              math.radians(FOE_HEADING_SPREAD_DEG))
        ic = Initial(
            "evader",
            ((bx, by), (0.0, 0.0)),              # red is placed relative to blue
            (math.degrees(psi), math.degrees(foe_psi)),
            (rng.uniform(OWN_SPEED_LO_KT, OWN_SPEED_HI_KT), FOE_SPEED_KT),
            (ALT_FT, ALT_FT),
        )
        return ic, Evader(rng)
