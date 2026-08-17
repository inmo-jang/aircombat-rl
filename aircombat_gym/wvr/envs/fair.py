"""`AirCombat/FairFight-v0` -- the even merge.  Nobody is given anything.

    import gymnasium as gym, aircombat_gym.wvr.envs
    env = gym.make("AirCombat/FairFight-v0")

`AdvantagedFight` with the initiative taken away.  Same opponent, same gun, same
clock, same closed vertical.  What changes is where the fight starts.

      red             blue
       ^                |
       |                |
       *  <- 10,000 m ->*
                        |
                        v

    red on the west looking north, blue on the east looking south

Abeam, 10,000 m apart, opposite headings, each on the other's 3/9 line.  The
picture is unchanged by rotating it 180 degrees about the midpoint, so the two
aircraft see the same problem: neither starts inside the other's turn, and
neither is looking at the other.  Both have to convert before they can shoot.

**Both aircraft start at the same speed**, 450 kt give or take 10 %, and each
nose is jittered 10 degrees either side of abeam.  Equal energy is the point:
drawing a speed per side would decide the fight before it began.  The throttle
is open from the first step, so what a pilot does with that energy is still
entirely up to it.

The separation was 2,000 m and is now 10,000 m, because at 2,000 m nothing
happened: 200 engagements, every pairing, 200 timeouts.  Two aircraft each
drawing a 1,070 m radius circle from 2,000 m apart never enter each other's
turn, so it is a Lufbery from the merge and no amount of clock resolves it.  At
10,000 m they have to convert before they arrive -- five quadrant transitions in
the first minute against none at 2 km -- and damage starts being traded.

The variety is deliberately thin, and the cost is worth stating.  A policy's
observation is entirely relative, so translating or rotating this setup produces
a byte-identical input; what is left is 20 degrees of jitter and 90 kt of common
speed.  A policy here learns one fight very well.  `AdvantagedFight`, whose
bearing and blue heading are drawn over the full circle, is where variety lives.

**Two copies of one script will not resolve this.**  With the geometry exactly
fixed, `ace`, `lead` and `pursuit` all score 0.00 with 60 timeouts out of 60:
identical aircraft flown identically mirror each other forever, which is a
Lufbery and not a bug.  A random policy still loses 60 out of 60, so kills are
reachable -- it is symmetry, not the weapon, that prevents them.  So the
scripted baselines calibrate nothing here; the reference that means something is
a *learned* policy against `ace`, because those two are not the same pilot.

Why this and not the head-on merge people picture when they hear "even": a
nose-to-nose pass ends in mutual destruction 29 times in 30.  `flat_damage`
overrides the aspect factor that would make a head-on shot weak, so both
aircraft solve each other at the same moment.  Abeam avoids that without
favouring anyone.
"""
from __future__ import annotations

from ..baselines import Ace
from .base import DuelEnv, Initial

ALT_FT = 20_000.0
SEPARATION_M = 10_000.0
HEADING_JITTER_DEG = 10.0       # each nose, either side of its abeam heading
MID_SPEED_KT = 450.0
SPEED_JITTER = 0.10             # +-10 % of it, and *both* aircraft get it


class FairFightEnv(DuelEnv):
    """`Discrete(9)` = heading 3 x speed 3, or `Box(2)`.

    A sibling of `AdvantagedFightEnv`, not a subclass of it: the two share the
    weapon, the clock and the four-outcome `verdict`, and differ only in the
    starting picture.  An even fight is not a kind of uneven one.
    """

    def sample(self, rng):
        j = HEADING_JITTER_DEG
        # One speed, both aircraft.  Equal energy is what makes the merge even;
        # drawing it per side would hand somebody the fight before it started,
        # and the throttle is open from the first step either way.
        v = MID_SPEED_KT * (1.0 + rng.uniform(-SPEED_JITTER, SPEED_JITTER))
        ic = Initial(
            "neutral",
            # Centred on the origin, not hung off it.  Placing red at (0, 0)
            # and blue at (0, SEPARATION_M) reads as symmetric and is not: the
            # 180 degree rotation that maps one aircraft onto the other is about
            # the midpoint, and anything anchored to the origin -- the arena
            # circle, the lat/lon the positions are projected onto -- does not
            # come along.  Measured with `ace` in both seats over 300
            # engagements, red was shot down 0.28 against 0.10 shooting, and
            # swapping the two starting conditions flipped the sign exactly.
            # East/west, not north/south.  Positions are projected onto a
            # globe, so two aircraft placed 5 km north and 5 km south of the
            # origin sit at different latitudes and are not the same problem --
            # measured with the jitter zeroed so the start was an exact mirror,
            # `ace` in both seats lost 100 times out of 100 as red and won 100
            # as blue, with the health difference at -0.017 and +0.017.  Laid
            # out east/west the two share a latitude, and the same test comes
            # out at 1.00 mutual, health difference 0.000, unchanged when the
            # seats are swapped.  The merge is the same picture turned a quarter
            # turn: red west looking north, blue east looking south, each still
            # on the other's 3/9 line.
            ((-0.5 * SEPARATION_M, 0.0), (+0.5 * SEPARATION_M, 0.0)),
            (0.0 + rng.uniform(-j, j),           # red north, blue south
             180.0 + rng.uniform(-j, j)),
            (v, v),
            (ALT_FT, ALT_FT),
        )
        return ic, Ace()
