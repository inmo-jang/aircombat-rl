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
    evader    cruises, then breaks hard in a random direction when you close.
              The first opponent that answers back, and the first a memorised
              open-loop answer cannot beat.

Plain Python on a dict, so they run without torch.
"""
from __future__ import annotations

import math

import numpy as _np

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
    """The interface `TaskEnv` drives for the seat the learner is not in.

    `act(info, obs)` gets both halves of what a pilot could know: the derived
    match material, and the raw state vector as that seat sees it.  Every bot
    here ignores `obs` -- they are written against geometry, and the vector
    would only make them longer.  It is in the signature for `PolicyDriver`,
    which cannot work without it: `info` carries no positions, so the
    39-channel observation cannot be rebuilt from it.

    The env passes both rather than letting the driver reach back for them, so
    that "which snapshot does each pilot decide from" is settled in one place.
    That question has bitten once already -- calling `observe()` a second time
    inside `step` gave the opponent a quarter-step-fresher view of health, and
    `ace` in both seats then lost as red 100 times out of 100.
    """

    name = "bot"

    def reset(self) -> None:
        pass

    def act(self, info=None, obs=None) -> tuple[float, float, float]:
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

    def act(self, info=None, obs=None) -> tuple[float, float, float]:
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

    def act(self, info, obs=None):
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

    def act(self, info, obs=None):
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
        of elevation, which is outside the 30 deg cone.

    `pursuit` and `lead` are kept as the naive rungs beneath this.  A learner
    that beats them but not this one has found the obvious answer only.
    """

    name = "ace"
    CLOSE_M = 900.0

    def act(self, info, obs=None):
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
    """Cruises straight, and breaks hard in a random direction when threatened.

    Rung 2, and the property that earns it the rung is not difficulty -- it is
    that it *answers back*.  A policy that memorised one approach cannot beat
    something whose next move depends on where the attacker is.  The 2D version
    measured exactly this: the same algorithm went from 2 kills out of 12
    against a scripted circle to 10 out of 12 against a reactive opponent,
    because the circle could be memorised and the reaction could not.

    Two things are random, and both come from the caller's generator rather
    than from `random`:

        which way it breaks     otherwise the answer is "always cut left"
        how long it holds it    otherwise the answer is a stopwatch

    Passing the generator in is not fastidiousness.  A bot that draws from the
    module-level `random` puts an unseeded stream inside an env whose whole
    evaluation protocol is three disjoint seed bands, and every graded number
    stops being reproducible.

    Purely horizontal: the vertical is closed in the tasks this flies in, and a
    break that traded height for turn rate would be commanding a channel the
    learner is not allowed to answer on.
    """

    name = "evader"
    THREAT_R = 2500.0
    # `aa` is the angle between the line of sight and *his* nose, so it reads 0
    # when he is pointed away from me and pi when he is pointed at me.  This is
    # the same convention the damage model uses -- `cos(aa)` peaks at his six --
    # and the version of this bot that shipped had the test the other way round,
    # so it broke only when the attacker was looking somewhere else.  It fired
    # 0.1 times a match instead of 3.
    THREAT_AA = math.radians(120.0)     # his nose within 60 deg of me
    # 4-8 s, not 2-4.  Measured on the ace-minus-pursuit gap, which is what a
    # task is worth: a longer break bleeds more of blue's energy, and `ace`
    # converts that while `pursuit` does not.  Doubling the break took the gap
    # from +0.10 to +0.17, and pairing it with a 240 s clock took it to +0.22.
    BREAK_LO_S, BREAK_HI_S = 4.0, 8.0
    RECOVER_S = 1.0                     # wings level before it will break again

    def __init__(self, rng=None):
        self.rng = rng if rng is not None else _np.random.default_rng()

    def reset(self):
        self.breaking = 0
        self.cooldown = 0
        self.side = 1.0

    def act(self, info, obs=None):
        threatened = (info["range"] < self.THREAT_R
                      and info["aa"] > self.THREAT_AA)
        if self.breaking > 0:
            self.breaking -= 1
            return (self.side * TURN, 0.0, 0.0)
        if self.cooldown > 0:
            self.cooldown -= 1
        elif threatened:
            self.breaking = int(DECISION_HZ * self.rng.uniform(
                self.BREAK_LO_S, self.BREAK_HI_S))
            self.cooldown = int(DECISION_HZ * self.RECOVER_S)
            self.side = 1.0 if self.rng.random() < 0.5 else -1.0
            return (self.side * TURN, 0.0, 0.0)
        # not threatened: run, and keep the energy up to make the run worth it
        return (0.0, DV if info["own_speed"] < CORNER_KT else 0.0, 0.0)


class PolicyDriver(Bot):
    """A trained policy in the seat a bot would otherwise fly.

    This is what a policy-versus-policy match is made of.  Every other bot
    here is hand-written and reads `info`; this one reads `obs`, the same
    39-channel vector the learner in the other seat gets, encoded from this
    seat's point of view.

        env = AdvantagedFightEnv(seat="red")   # student A drives the gym
        foe = PolicyDriver(B.predict, env)     # student B flies blue

    `predict` takes the raw 39 channels and returns whatever `env.decode`
    accepts -- an index for a discrete space, a vector for a continuous one.
    Wrapping the submission's own observation transform is the submission's
    business, exactly as it is when it plays the learner's seat.
    """

    name = "policy"

    def __init__(self, predict, env, transform=None) -> None:
        self.predict = predict
        self.decode = env.decode
        self.transform = transform

    def act(self, info=None, obs=None):
        if obs is None:
            raise ValueError("PolicyDriver needs the observation; the env "
                             "passes it as the second argument to act()")
        x = obs if self.transform is None else self.transform(obs)
        return self.decode(self.predict(x))


LADDER = (Circler, Pursuit, Lead, Ace, Evader)
BY_NAME = {c.name: c for c in LADDER}
