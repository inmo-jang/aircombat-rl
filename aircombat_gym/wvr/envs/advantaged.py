"""`AirCombat/AdvantagedFight-v0` -- a duel you enter from the front quarter.

    import gymnasium as gym, aircombat_gym.wvr.envs
    env = gym.make("AirCombat/AdvantagedFight-v0")

Same aircraft, same altitude, same closed vertical, same `Discrete(9)`, same
clock.  Blue is `baselines.Ace`, and it carries **the same gun on the same
terms**: 30 deg cone, 1.0 s lock, 0.33/s flat damage.  Nothing is
handicapped.

Three things follow from a target that shoots back, and all three change how a
reward has to be written:

  red can die.  The first task that reaches `verdict`'s losing branch on
                purpose.  A reward carried over from `Circular` or `Evader` is
                not wrong here so much as blind -- it cannot see the half of the
                episode that kills you.
  the clock is a draw, not a loss.  Two aircraft that never solve each other
                have not lost to each other.  `verdict` still reports `timeout`
                so Gymnasium truncates rather than terminates.
  simultaneous death is a draw.  Symmetric weapons make it reachable, and
                awarding it to red because `verdict` happened to test blue first
                would be a scoring artefact.

`Ace` is 2D here without being modified: it commands altitude only through
`_match_alt`, which returns zero while both aircraft are pinned at 20,000 ft.

**Red starts pointed at blue, within 90 deg, exactly as in `Circular` and
`Evader`; blue's heading is free.**  Red therefore holds the initiative on
average, and that is the rung.  The even merge is `FairFight`.

The tilt is in the name because it is large enough to plan around.  Measured
over 2,000 draws of `sample()`, and against `FairFight` for scale:

                        red aim off LOS    blue aim off LOS
    AdvantagedFight          44.4 deg           90.3 deg
    FairFight                89.9 deg           90.0 deg

    starting quadrant, red's view:  offensive 0.41  neutral 0.32
                                    head-on   0.16  defensive 0.12

Offensive outnumbers defensive 3.4 to 1, and it shows in the result: `ace`
against a copy of itself scores 0.35 kills to 0.23 deaths over 60 engagements,
+0.117 to red.  Opening red's cone to the full circle takes that to -0.067,
which is even -- and takes the task with it, for the reason below.

That asymmetry is not a concession, it is the measurement.  With both headings
free, `ace` against a copy of itself wins 0.72 of the engagements it enters from
60 deg or more inside the opponent's nose and **0.00 of the 27 it enters from
outside** -- 87 % of episodes are decided before anyone manoeuvres.  `Circular`
and `Evader` never had that problem because both point red at the target; this
task did not, and it was the only difference.

Speeds are 300-600 kt for both.  `Circular` and `Evader` pin the *target* at
325 kt so a fleeing aircraft stays catchable; nothing flees here.

Four other openings were measured and do not work:

  head-on merge     29 in 30 end in mutual destruction -- `flat_damage`
                    overrides the aspect factor, so a nose-to-nose pass is as
                    lethal as the six (see `fair.py`).
  honest damage     `flat_damage=None` kills nobody: 30 timeouts out of 30.  The
                    three-factor rate is too low to finish inside the clock.
  symmetric offset  both noses 90 deg off the line of sight, same rotational
                    sense, is a Lufbery: 30 timeouts.
  asymmetric offset resolves, at the cost of being rigged -- rotating only blue
                    gave red 17 wins and 0 losses.

Free headings, kept for `FairFight`, give a distribution that is neutral without
any single episode having to be: `ace` against a copy of itself goes 11-10 over
40.  What they do not give is a task where flying well changes the answer.
"""
from __future__ import annotations

import math

from ..baselines import Ace
from .base import DuelEnv, Initial

ALT_FT = 20_000.0                       # both aircraft, fixed
# Drawn over 4-10 km rather than pinned, because a policy trained at one range
# only works at that range.  The global policy learned at 2-4 km scores 0.65 at
# 10 km and collapses to a 0.97 mutual rate at 6 km -- just outside what it saw.
# `ace`, which learned nothing, has no such hole.  The band covers every
# distance the sweep measured, so "which range do we ship" stops being a
# question the policy can get wrong.
RANGE_LO_M, RANGE_HI_M = 4_000.0, 10_000.0
HEADING_CONE_DEG = 90.0                 # red's heading, either side of the LOS
FOE_HEADING_SPREAD_DEG = 180.0          # blue's is free
SPEED_LO_KT, SPEED_HI_KT = 300.0, 600.0

# A survivor left with this much health or less did not win, it traded.
#
# Two mirror-image aircraft are integrated separately and floating point is not
# symmetric, so a perfectly even fight does not stay perfectly even: measured on
# 04 with the jitter zeroed, red and blue took their first damage on the *same*
# step at every distance, then traded for another 43 seconds, and at 8 km one of
# them came out at 0.0100 health while the other reached zero.  The environment
# called that a clean kill 120 times out of 120.  It is not a clean kill; it is
# rounding, and one hundredth of a health bar is smaller than a single step's
# damage (0.0165 at `flat_damage=0.33`).
#


class AdvantagedFightEnv(DuelEnv):
    """`Discrete(9)` = heading 3 x speed 3, or `Box(2)`."""

    def sample(self, rng):
        """Red west of blue, both on the same latitude, headings carry the rest.

        The bearing used to be drawn over the full circle and applied to the
        *positions*, which put the two aircraft at different latitudes on every
        draw but one.  Positions are projected onto a globe, so that is not a
        rotation of one problem -- it is a different problem each time, and a
        biased one: with the geometry made an exact mirror, `ace` in both seats
        lost 100 engagements out of 100 from the southern seat and won 100 from
        the northern.

        Nothing is lost by fixing the axis.  A policy's observation is entirely
        relative -- positions differenced, then rotated into its own heading
        frame -- so rotating the whole picture produces a byte-identical input.
        The bearing therefore moves from the positions into the headings, which
        is where it was doing its work anyway: what varies is red's angle off
        the line of sight and blue's, and both are still drawn over exactly the
        ranges they were.
        """
        r = rng.uniform(RANGE_LO_M, RANGE_HI_M)
        # Red west, blue east, the line of sight due east: heading 90 degrees.
        toward_foe = 0.5 * math.pi
        psi = toward_foe + rng.uniform(-math.radians(HEADING_CONE_DEG),
                                       math.radians(HEADING_CONE_DEG))
        foe_psi = rng.uniform(-math.radians(FOE_HEADING_SPREAD_DEG),
                              math.radians(FOE_HEADING_SPREAD_DEG))
        ic = Initial(
            "ace",
            ((-0.5 * r, 0.0), (+0.5 * r, 0.0)),
            (math.degrees(psi), math.degrees(foe_psi)),
            (rng.uniform(SPEED_LO_KT, SPEED_HI_KT),
             rng.uniform(SPEED_LO_KT, SPEED_HI_KT)),
            (ALT_FT, ALT_FT),
        )
        return ic, Ace()
