"""Scripted opponents for the gun.

A bot is anything with `reset()` and `act(info) -> (d_heading, d_speed, d_alt)`
-- the same three deltas a policy emits.  `info` is one side's dict out of
`envs.base.Combat.observe()`, not the observation vector: these are the
referee's aircraft, not entrants, so they may read the state directly.

They form a ladder, and the rungs are about *what the learner has to discover*:

    circler   holds a turn and ignores you.  Anything that can point and close
              scores; difficulty is the turn rate.
    pursuit   points at you and pulls.  The obvious strategy, and the one a
              learner has to beat before it can claim anything.
    lead      aims where you will be.  Same effort as pursuit, so a score gap
              between them is a difference in aiming and nothing else.
    ace       lead pursuit plus closure control.  The strongest reference, and
              what the assignment's baselines are measured against.
    evader    watches its six and breaks.  The first opponent that answers back,
              and the first a memorised open-loop answer cannot beat.

Plain Python on a dict, so they run without torch.
"""
from __future__ import annotations

import math

from ..core.aircraft import DECISION_HZ
from .actions import DELTA_ALT_FT, DELTA_HEADING_DEG, DELTA_SPEED_KT

TURN = DELTA_HEADING_DEG[-1]        # 30 deg -- the only turn magnitude there is
DV = DELTA_SPEED_KT[-1]             # 20 kt
DALT = DELTA_ALT_FT[-1]             # 1000 ft
CORNER_KT = 500.0


def _snap(want: float, step: float) -> float:
    """Nearest of (-step, 0, +step).  The action grid is three-valued."""
    if want > 0.5 * step:
        return step
    if want < -0.5 * step:
        return -step
    return 0.0


class Bot:
    """The interface `Combat` drives, plus the two steering helpers most bots
    want.  `act` may ignore `info` -- a target drone does."""

    name = "bot"

    def reset(self) -> None:
        pass

    def act(self, info=None) -> tuple[float, float, float]:
        raise NotImplementedError

    @staticmethod
    def _turn(ata_signed: float, gain: float = 4.0) -> float:
        """Steer toward the line of sight.  Snapped to the action grid."""
        return _snap(math.degrees(ata_signed) * gain, TURN)

    @staticmethod
    def _match_alt(info, gain: float = 1.0) -> float:
        return _snap(info["alt_diff"] * gain, DALT)


class Circler(Bot):
    """Constant turn, constant height, constant speed.  No evasion, no fire.

    The easiest target there is, and that is the point -- rung 1 of the ladder,
    the case where anything that can point and close should score.  When it
    stops being trivially catchable, something is wrong with the weapon or the
    geometry rather than with the pilot.

    `turn_rate_deg_s` is an **average**, not an instantaneous rate.  The action
    grid has one turn magnitude, so a slow turn is that 30 deg step commanded
    rarely: the target rolls in, turns, rolls out and flies straight until the
    next one.  Measured over 120 s -- 1 deg/s gives 4 commands and 62 % of the
    time wings-level, 5 deg/s gives 20 and 6 %.  A polygon, not a circle, and at
    1 deg/s a policy is mostly learning to attack a straight target.

    The rate is carried rather than divided into a duty cycle, which is the bug
    this replaced: commanding the step every `period`-th decision demands
    600/period deg/s against an airframe that delivers 12, so every period ever
    tried saturated and measured the same 6.9 deg/s.
    """

    name = "circler"

    def __init__(self, turn_rate_deg_s: float = 4.0) -> None:
        self.rate = float(turn_rate_deg_s)

    def reset(self) -> None:
        self._carry = 0.0

    def act(self, info=None) -> tuple[float, float, float]:
        self._carry += self.rate / DECISION_HZ
        if abs(self._carry) >= abs(TURN):
            self._carry -= math.copysign(TURN, self.rate)
            return math.copysign(TURN, self.rate), 0.0, 0.0
        return 0.0, 0.0, 0.0                # turn, hold speed, hold altitude


class Pursuit(Bot):
    """Point at him, pull as hard as possible, ask for speed, match height.

    The obvious strategy and the one to beat.  If a learner cannot beat this it
    has not learned the task; if beating it is all it can do, it has learned a
    rate contest and nothing about energy.
    """

    name = "pursuit"

    def act(self, info):
        return (self._turn(info["ata_signed"]),
                DV if info["own_speed"] < CORNER_KT else 0.0,
                self._match_alt(info))


class Lead(Bot):
    """Lead pursuit: aim where he is going, not where he is.

    Identical effort to `pursuit`; only the aim point differs, so any score gap
    between them is a difference in aiming policy and nothing else.  `ata_lead`
    is unsigned, so the sign comes from the bearing.
    """

    name = "lead"

    def act(self, info):
        s = math.copysign(1.0, info["ata_signed"])
        return (self._turn(info["ata_lead"] * s),
                DV if info["own_speed"] < CORNER_KT else 0.0,
                self._match_alt(info))


class Ace(Bot):
    """Lead pursuit with closure control.  The strongest hand-written reference.

    Three things separate it from `lead`, and all three were measured:

      * it steers to `lead_signed`, the signed bearing of the aim point, not to
        an unsigned lead angle wearing the sign of the target bearing.  Those
        differ exactly when the pipper crosses the nose, which is the moment the
        shot exists.
      * it manages closure.  Holding 500 kt behind a 325 kt target flies a wider
        circle than the target's and the solution never settles; slowing inside
        gun range is what lets the nose stay on.
      * it matches altitude aggressively.  A 1,000 ft split at 800 m is 35 deg
        of elevation, which is well outside a 15 deg cone.

    `pursuit` and `lead` are kept as the naive rungs beneath this.  A learner
    that beats them but not this one has found the obvious answer only.
    """

    name = "ace"
    CLOSE_M = 900.0

    def act(self, info):
        r = info["range"]
        turn = self._turn(info["lead_signed"], 4.0)
        if r > self.CLOSE_M:
            dv = DV                                   # run him down
        elif info["range_rate"] > -20.0 and r > 400.0:
            dv = 0.0                                  # stabilised in the saddle
        else:
            dv = -DV                                  # closing too fast, overshoot
        return turn, dv, self._match_alt(info, 2.0)


class Evader(Bot):
    """Checks its six and breaks; otherwise it hunts.

    Rung 2.  The important property is not that it is hard, it is that it
    *answers back*: an open-loop policy that memorised one approach cannot beat
    something whose next move depends on where the attacker is.  The 2D version
    measured exactly this -- the same algorithm went from 2 kills out of 12
    against a scripted circle to 10 out of 12 against a reactive opponent,
    because the circle could be memorised and the reaction could not.

    Break turn plus a descent: trading height for turn rate is the right answer
    when someone is behind you, and it makes the vertical part of the fight.
    """

    name = "evader"
    THREAT_R = 2500.0
    THREAT_AA = math.radians(60.0)      # he is inside my rear quarter

    def reset(self):
        self.breaking = 0
        self.side = 1.0

    def act(self, info):
        # `aa` is measured from *his* nose to me, so a small aa means he is
        # pointing at me.  That plus close range is the definition of trouble.
        threatened = info["range"] < self.THREAT_R and info["aa"] < self.THREAT_AA
        if threatened and self.breaking <= 0:
            self.breaking = 60                       # 3 s of committed break
            self.side = -math.copysign(1.0, info["ata_signed"] or 1.0)
        if self.breaking > 0:
            self.breaking -= 1
            return (self.side * TURN, 0.0, -DALT)    # break and unload downhill
        return (self._turn(info["ata_signed"]),
                DV if info["own_speed"] < CORNER_KT else 0.0,
                self._match_alt(info))


LADDER = (Circler, Pursuit, Lead, Ace, Evader)
ALL = LADDER
BY_NAME = {c.name: c for c in ALL}
