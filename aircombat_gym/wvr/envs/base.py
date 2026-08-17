"""The two aircraft, the weapon bookkeeping, and the Gymnasium surface.

Two classes, and the split between them is the one that matters:

    Combat    runs both aircraft and *records* what the weapon did -- health,
              how long each side has held a firing solution, the clock.  It
              decides nothing.
    TaskEnv   the Gymnasium environment.  It decides: `verdict()` says what ends
              a match and who won, and each task overrides it as far as it
              needs to.

Judging used to happen in both places -- a `Report` built here and then ignored
by the env, which re-derived the same answer from health and time.  Two judges
drift; the timeout rule already had.

**No reward.**  `step` returns 0.0, always.  The material for a reward goes out
in `info` and the student writes the function -- `tests/test_envs.py` walks the
package AST to keep it that way.

A task is one file next to this one.  It supplies `sample()`, sets a few
class attributes, and overrides `verdict()` if its difficulty needs it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .. import obs as O
from ...core import aircraft as _ac
from ...core.aircraft import Aircraft
from ...core.envelope import ALT_MAX_FT, ALT_MIN_FT, H0_FT
from .. import actions as act
from ..engagement import MUZZLE_MS, WEZ_ATA_DEG, WEZ_R_MAX, WEZ_R_MIN, look

SIDES = ("red", "blue")

# Horizontal radius of the arena, centred on the projection origin (Seoul).
#
# 50 km is chosen to be *out of reach without being infinite*.  An F-16 flying
# straight for the whole 120 s clock covers 27.8 km at 450 kt and 40.1 km at the
# 650 kt ceiling, so nothing reaches the edge in normal play and the boundary
# never decides a match by surprise.  It is still a number rather than a
# disabled rule, which matters twice: a runaway is bounded, and the local
# tangent plane stays honest.  `x` and `y` come from an equirectangular
# projection with `cos(lat0)` fixed at the origin, so the east-west scale error
# grows with distance north -- 0.5 % at 40 km, but 6.3 % at 500 km.  A 500 km
# arena would be an arena the projection cannot describe.
#
# It replaces a 10 km radius that no task could honour: `Circular`'s target
# orbits out to 9.6 km on its own.
ARENA_R = 50_000.0

# Leaving the measured altitude band means `n_max(V, h)` is extrapolated off the
# end of the table -- the aircraft would be flying on numbers nobody measured.
ALT_FLOOR_FT = ALT_MIN_FT - 500.0
ALT_CEIL_FT = ALT_MAX_FT + 500.0

# Action channels.  A task opens the ones it wants; the closed ones are dropped
# from the space entirely rather than being present and ignored, because an
# action a policy can take and that does nothing is pure exploration tax.
HEADING, SPEED, ALTITUDE = "heading", "speed", "altitude"
_GRID = {HEADING: act.DELTA_HEADING_DEG,
         SPEED: act.DELTA_SPEED_KT,
         ALTITUDE: act.DELTA_ALT_FT}
_ORDER = (HEADING, SPEED, ALTITUDE)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi



@dataclass
class Initial:
    """A starting geometry.  Positions in metres, altitudes in feet."""

    name: str
    xy: tuple[tuple[float, float], tuple[float, float]]
    psi_deg: tuple[float, float]
    v_kt: tuple[float, float]
    alt_ft: tuple[float, float] = (H0_FT, H0_FT)


# =============================================================================
# Combat -- runs the fight, records what happened, judges nothing
# =============================================================================

class Combat:
    """Two aircraft, one weapon model, one clock.

    Both guns are modelled.  A learner needs the other barrel from the first
    step, because defence is not something to bolt on afterwards -- it changes
    the observation (`own_health` stops being a constant) and every reward
    written against it.

    Ported from the 2D version on 2026-08-10.  Three things had to change, and
    the first is the kind of bug that hides until the day it does not:

      * `_kin` had `vz = -h_dot` while `look()` measures `dz` upward-positive.
        With the vertical locked and `h_dot` near zero it was invisible; open
        the vertical and it inverts every closing rate and lead angle.  The
        horizontal component is `v*cos(gamma)`, not `v`, for the same reason.
      * the two aircraft no longer share one altitude.
      * the arena is a cylinder: a horizontal radius plus the measured altitude
        band.
    """

    def __init__(self, t_max: float, track_lock: float,
                 flat_damage: float | None = None,
                 wez_cone_deg: float = WEZ_ATA_DEG,
                 arena_m: float = ARENA_R,
                 armed: tuple[str, ...] = SIDES) -> None:
        """`flat_damage` replaces the three-factor damage model with a constant.

        The default model scales damage by aim error, range and aspect, which is
        honest for a weapon and wrong for a first assignment: a student cannot
        work out how long to hold the shot, because the answer depends on three
        things at once.  `flat_damage=r` makes it exactly `1/r` seconds of
        tracking after the lock, and nothing else.

        `wez_cone_deg` and `flat_damage` are arguments rather than module
        constants on purpose: `SubprocVecEnv(start_method="spawn")` re-imports
        the module in each worker, so a task that reassigns a global gets the
        default everywhere it matters.  That mistake made two "different"
        training runs byte-for-byte identical for 1,592 episodes.

        `armed` names the sides that carry one.  An aircraft left out of it has
        no gun at all: no solution, no damage, and `under_track` reads 0 for
        whoever it is chasing, which is what "unarmed" should mean to a policy.
        A task whose opponent is described as a target has to say so here --
        both barrels fire by default, and a drone that has never aimed at
        anything still lands 0.16 s of solution per match by flying into one.
        """
        self.t_max = t_max
        self.track_lock = track_lock
        self.flat_damage = flat_damage
        self.wez_cone_deg = wez_cone_deg
        self.arena_m = arena_m
        self.armed = tuple(armed)
        self.ac = {s: Aircraft(h0_ft=H0_FT) for s in SIDES}
        self._zero()

    def _zero(self) -> None:
        self.health = {s: 1.0 for s in SIDES}
        self.wez_time = {s: 0.0 for s in SIDES}
        self.track = {s: 0.0 for s in SIDES}
        self.damage = {s: 0.0 for s in SIDES}
        self.first_wez = {s: None for s in SIDES}
        self.t = 0.0
        self.min_range = 1e9

    def reset(self, ic: Initial) -> dict:
        for s, k in zip(SIDES, (0, 1)):
            # h0_ft is what a zero altitude delta freezes onto, so it has to be
            # this episode's start rather than the class default
            self.ac[s].h0_ft = ic.alt_ft[k]
            self.ac[s].autopilot.h0_ft = ic.alt_ft[k]
            self.ac[s].reset(x=ic.xy[k][0], y=ic.xy[k][1],
                             psi=math.radians(ic.psi_deg[k]), v_kt=ic.v_kt[k])
        self._zero()
        return self.observe()

    def observe(self) -> dict:
        st = {s: self.ac[s].state for s in SIDES}
        c = self.wez_cone_deg
        eng = {"red": look(_kin(st["red"]), _kin(st["blue"]), c),
               "blue": look(_kin(st["blue"]), _kin(st["red"]), c)}
        out = {}
        for s in SIDES:
            o = "blue" if s == "red" else "red"
            e = eng[s]
            # Signed bearing to the *aim point*, which is what steering wants.
            # `ata_lead` is unsigned, and borrowing the sign of `ata_signed` is
            # wrong exactly when the lead point has crossed the nose -- which is
            # when the shot is about to be there.
            lead_signed = wrap_pi(math.atan2(e.aim_dx, e.aim_dy) - st[s].psi)
            out[s] = dict(
                ata=e.ata, ata_signed=e.ata_signed, ata_lead=e.ata_lead,
                lead_signed=lead_signed, aa=e.aa,
                range=e.r, range_rate=e.r_dot, in_wez=e.in_wez,
                # Whether *he* has the solution on me.  Derivable from his nose
                # and the range, but the panel hands out my half for free and an
                # asymmetric panel teaches an asymmetric policy -- twenty-one
                # reward functions in a row paid for own aim angle and none for
                # his.  It also makes a mirror test on the observation possible.
                opp_in_wez=eng[o].in_wez,
                damage_rate=e.damage_rate,
                own_speed=st[s].v_kt, opp_speed=st[o].v_kt,
                own_health=self.health[s], opp_health=self.health[o],
                dist_to_boundary=self.arena_m - math.hypot(st[s].x, st[s].y),
                opp_dist_to_boundary=(self.arena_m
                                      - math.hypot(st[o].x, st[o].y)),
                t=self.t, t_remaining=self.t_max - self.t,
                wez_time_own=self.wez_time[s], wez_time_opp=self.wez_time[o],
                track_time=self.track[s], under_track=self.track[o],
                own_alt=st[s].h_ft, alt_diff=st[o].h_ft - st[s].h_ft,
                bearing_center=math.atan2(-st[s].x, -st[s].y) - st[s].psi,
            )
        self._eng = eng
        return out

    def step(self, actions: dict) -> dict:
        """One decision step for both aircraft, then the weapon bookkeeping.

        Read this before designing a reward.  The streak resets and the health
        does not, so a solution held for 0.9 s is worth nothing however often it
        is repeated:

            in_wez:  track += dt;  track >= lock  ->  health -= rate * dt
            else:    track  = 0.0

        A side's action may be `None`, meaning "I have already advanced this
        aircraft myself -- just keep score for it".  That is for the keyboard
        bench, which drives its own aircraft three different ways (control
        surfaces, an action delta, or a frozen target) and cannot express two of
        them as deltas.  Before it existed the bench carried its own copy of the
        bookkeeping below, and the copy drifted: a 0.6 s lock against 1.0, a
        15 deg cone against 30, the three-factor damage model against a flat
        rate, and only one of the two guns modelled at all.
        """
        for s in SIDES:
            if actions[s] is not None:
                self.ac[s].step(*actions[s])
        dt = 1.0 / _ac.DECISION_HZ
        self.t += dt

        info = self.observe()
        for s in SIDES:
            o = "blue" if s == "red" else "red"
            e = self._eng[s]
            if s not in self.armed:
                continue
            if e.in_wez:
                if self.first_wez[s] is None:
                    self.first_wez[s] = self.t
                self.wez_time[s] += dt
                self.track[s] += dt
                if self.track[s] >= self.track_lock:
                    d = (self.flat_damage if self.flat_damage is not None
                         else e.damage_rate) * dt
                    self.damage[s] += d
                    self.health[o] = max(0.0, self.health[o] - d)
            else:
                self.track[s] = 0.0        # solution broken, start again

        # The snapshot above was taken before any of that damage landed, because
        # the damage needs `self._eng`, which `observe()` is what fills in.  The
        # geometry in it is correct -- nothing here moves an aircraft -- but the
        # health is a tick stale, and a tick is 0.0165 of a health bar at
        # `flat_damage=0.33`.  Measured on 04 at 8 km, `info` reported the
        # survivor on 0.0265 while `verdict`, which reads `self.health`, saw
        # 0.0100: on opposite sides of the 0.02 line that decides whether the
        # engagement was a kill or a trade.
        #
        # Patched rather than re-observed: `observe()` recomputes both
        # engagements from aircraft state that has not changed, so calling it
        # again would cost a full pass to return the same geometry.  Only the
        # four health numbers moved.
        for s in SIDES:
            o = "blue" if s == "red" else "red"
            info[s]["own_health"] = self.health[s]
            info[s]["opp_health"] = self.health[o]

        self.min_range = min(self.min_range, self._eng["red"].r)
        return info

    # --- facts a verdict may want.  Still not a judgement. -------------------

    def out_of_bounds(self, side: str) -> bool:
        """Outside the cylinder, or off the measured envelope.

        **Nothing calls this yet.**  No `verdict()` on the ladder ends a match
        for leaving, so today this is a fact nobody asks for.  It is kept, and
        kept correct, because the arena rule is deferred rather than rejected --
        but the silence is worth saying out loud: an unread `lose_on_exit`
        attribute and an inert `arena_m` both survived here for weeks precisely
        because dead code looks like working code.
        """
        st = self.ac[side].state
        return (math.hypot(st.x, st.y) > self.arena_m
                or not (ALT_FLOOR_FT <= st.h_ft <= ALT_CEIL_FT)
                or not st.flyable)


@dataclass
class _Kin:
    x: float; y: float; h: float
    psi: float; theta: float
    vx: float; vy: float; vz: float


def _kin(s) -> _Kin:
    """What `engagement.look` needs.  See `Combat` for the signs."""
    vh = s.v * math.cos(s.gamma)
    return _Kin(s.x, s.y, s.h, s.psi, s.theta,
                math.sin(s.track) * vh, math.cos(s.track) * vh, s.h_dot)


# =============================================================================
# TaskEnv -- the Gymnasium surface, and the judge
# =============================================================================

class TaskEnv(gym.Env):
    """One learner against one opponent, either of them in either seat.

    An assignment subclasses this, implements `sample()`, and sets whichever
    class attributes differ.  `verdict()` is where its difficulty lives.

    **The learner sits in `seat` and everything is written relative to it.**
    It used to be `red`, spelled out in eleven places, and that made the last
    rung of the ladder impossible: a student-versus-student tournament has to be
    able to put either submission in either seat, and it has to be able to put a
    trained policy where the scripted bot goes.  Neither was reachable.

    `Combat` was already symmetric -- it runs both aircraft, scores both guns
    and builds both `info` dicts -- so the asymmetry was only ever in this
    class.  With `seat` a parameter, an identity falls out that is worth testing
    rather than assuming:

        seat="blue"  ==  seat="red" with the starting geometry swapped

    Those two must agree step for step.  The one time a seat bias was suspected
    it took eleven discarded hypotheses to find that the cause was the latitude
    projection, not the code path; this makes the code-path half a test.
    """

    metadata = {"render_modes": []}

    # --- what an assignment configures ---
    channels: tuple[str, ...] = (HEADING, SPEED)
    # Which side the learner flies.  Training against a scripted opponent
    # leaves it alone; a policy-versus-policy match sets it per engagement so
    # that both entrants fly both seats.
    seat: str = "red"
    # The snapshot both pilots decide from, carried across `step`.  A class
    # default so reaching `step` before `reset` fails the way Gymnasium says it
    # should rather than with an AttributeError from in here.
    _last: dict | None = None

    t_max: float = 120.0
    track_lock: float = 1.0
    flat_damage: float | None = 0.33
    wez_cone_deg: float = 30.0
    arena_m: float = ARENA_R
    # The rest of the sight.  Class attributes so an assignment can narrow the
    # band, and so a viewer can read the numbers in force off the env instead of
    # importing the weapon module and hoping they match.
    wez_r_min: float = WEZ_R_MIN
    wez_r_max: float = WEZ_R_MAX
    muzzle_ms: float = MUZZLE_MS
    # Both barrels fire on every rung of the ladder.  `Evader` used to
    # disarm its target; it is armed now, so that the only thing changing from
    # rung to rung is the opponent's behaviour and the geometry it starts in.
    # An evading target that never aims is harmless in practice and the uniform
    # rule is worth more than the guarantee.
    foe_armed: bool = True

    def __init__(self, action_mode: str = "discrete", seed: int | None = None,
                 seat: str | None = None, arena_m: float | None = None):
        super().__init__()
        if seat is not None:
            if seat not in SIDES:
                raise ValueError(f"seat {seat!r} not in {SIDES}")
            self.seat = seat
        if arena_m is not None:
            self.arena_m = arena_m
        self._grid = [_GRID[c] for c in _ORDER if c in self.channels]
        self._slot = [_ORDER.index(c) for c in _ORDER if c in self.channels]
        self._scale = np.array([abs(g[-1]) for g in self._grid], dtype=np.float32)

        self.set_action_mode(action_mode)
        self.observation_space = O.make_observation_space()

        self._combat = Combat(t_max=self.t_max, track_lock=self.track_lock,
                              flat_damage=self.flat_damage,
                              wez_cone_deg=self.wez_cone_deg,
                              arena_m=self.arena_m,
                              armed=SIDES if self.foe_armed else (self.seat,))
        self._rng = np.random.default_rng(seed)

    @property
    def foe_seat(self) -> str:
        return "blue" if self.seat == "red" else "red"

    def set_action_mode(self, mode: str) -> None:
        """Swap `Discrete(n)` for `Box(k)` and back, mid-episode.

        Only two things depend on the mode -- the string and the space -- so the
        aircraft, the weapon and the clock carry straight on.  That is what makes
        this worth having: a person can feel the same engagement under both
        action spaces without respawning into a different one, and the gap
        between them is the largest single effect measured on this ladder.

        **Not for training.**  A policy's output layer is bound to the space it
        was built with, and changing it underneath a learner is a silent
        mismatch rather than an error.  This exists for the hand-play harness.
        """
        if mode not in ("discrete", "continuous"):
            raise ValueError(f"action_mode {mode!r}")
        self.action_mode = mode
        if mode == "discrete":
            n = 1
            for g in self._grid:
                n *= len(g)
            self.action_space = gym.spaces.Discrete(n)
        else:
            self.action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(len(self._grid),), dtype=np.float32)

    # --- what an assignment supplies -----------------------------------------

    def sample(self, rng):
        """Draw one episode: `(Initial, opponent)`.  The opponent is pre-reset."""
        raise NotImplementedError

    def verdict(self, c: Combat):
        """`(ended, won, reason)`.  Override to change an assignment's rules.

        The default is the simplest thing that is a game: someone dies, or the
        clock runs out and the learner has not won.  Leaving the arena is *not*
        a loss here; a task that wants it to be overrides this and asks
        `c.out_of_bounds(...)`.

        Read relative to `self.seat`, so the same rule judges either side.
        """
        if c.health[self.foe_seat] <= 0.0:
            return True, True, "kill"
        if c.health[self.seat] <= 0.0:
            return True, False, "died"
        if c.t >= c.t_max:
            return True, False, "timeout"
        return False, None, ""

    # --- gym ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            super().reset(seed=seed)
        ic, foe = self.sample(self._rng)
        self._foe = foe
        self._foe.reset()
        info = self._combat.reset(ic)
        self._last = info
        return self._obs(info), self._info(info)

    def step(self, action):
        # Both pilots decide from the *same* snapshot, and it is the one the
        # learner was handed at the end of the previous step.
        #
        # This used to call `observe()` again here, which reads as "the opponent
        # decides from the state it can see now" and is a quarter of a step
        # fresher than what red saw: `Combat.step` takes its snapshot before
        # applying damage, so the dict red acts on carries last step's health
        # while a new `observe()` carries this step's.  Symmetric on paper,
        # decisive in practice -- with the jitter zeroed so that the two
        # aircraft start as exact mirror images, `ace` in both seats lost as red
        # 100 times out of 100 and never once traded.  Driving `Combat` directly
        # from one shared snapshot, the same fight times out every time.
        # The opponent is handed its observation by the env rather than building
        # one for itself, so that "which snapshot does each pilot see" stays a
        # decision made in exactly one place.  A scripted bot ignores `obs`; a
        # trained policy in this seat needs it, because `info` carries no
        # positions and the 39-channel vector cannot be rebuilt from it.
        snap = self._last if self._last is not None else self._combat.observe()
        foe_action = self._foe.act(snap[self.foe_seat],
                                   self.obs_for(self.foe_seat, snap))
        info = self._combat.step({self.seat: self.decode(action),
                                  self.foe_seat: foe_action})
        self._last = info
        ended, won, reason = self.verdict(self._combat)

        extra = self._info(info)
        if ended:
            extra.update(outcome=reason, won=won, t=self._combat.t,
                         wez_time=self._combat.wez_time[self.seat])
        # A loss and a timeout both end the episode; Gymnasium wants to know
        # which was the clock, because a value function must not bootstrap
        # through a real terminal but should through a truncation.
        truncated = ended and reason == "timeout"
        return (self._obs(info), 0.0, bool(ended and not truncated),
                bool(truncated), extra)

    # --- actions --------------------------------------------------------------

    def decode(self, action) -> tuple[float, float, float]:
        """Action -> the three deltas `Combat` wants.  Closed channels stay 0."""
        out = [0.0, 0.0, 0.0]
        if self.action_mode == "discrete":
            a = int(action)
            if not 0 <= a < self.action_space.n:
                raise ValueError(f"action {a} outside {self.action_space}")
            for slot, g in zip(reversed(self._slot), reversed(self._grid)):
                a, i = divmod(a, len(g))
                out[slot] = g[i]
        else:
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            for slot, v in zip(self._slot, (a * self._scale).tolist()):
                out[slot] = v
        return tuple(out)

    def encode_action(self, deltas):
        """The inverse of `decode`: three deltas -> an action.

        Lets a scripted bot be scored through the same surface a policy uses,
        which is the only way a baseline number means anything -- a baseline
        measured on a different code path is not a baseline.
        """
        vals = [deltas[s] for s in self._slot]
        if self.action_mode == "continuous":
            return np.clip(np.asarray(vals, dtype=np.float32) / self._scale,
                           -1.0, 1.0)
        a = 0
        for g, v in zip(self._grid, vals):
            a = a * len(g) + min(range(len(g)), key=lambda i: abs(g[i] - v))
        return a

    # --- internals ------------------------------------------------------------

    def obs_for(self, side: str, info: dict) -> np.ndarray:
        """The state vector as `side` sees it.  Public, because the tournament
        needs the opponent's copy and `info` alone cannot rebuild it."""
        other = "blue" if side == "red" else "red"
        return O.encode(self._combat.ac[side].state,
                        self._combat.ac[other].state, info[side])

    def _obs(self, info: dict) -> np.ndarray:
        return self.obs_for(self.seat, info)

    def _info(self, info: dict) -> dict:
        """The material a reward is built from.  Derived geometry lives here,
        never in the observation -- see `wvr/obs.py`."""
        i = info[self.seat]
        return {k: i[k] for k in
                ("range", "range_rate", "ata", "ata_signed", "ata_lead",
                 "lead_signed", "aa", "in_wez", "opp_in_wez", "track_time",
                 "under_track", "damage_rate", "own_health", "opp_health",
                 "own_speed", "opp_speed", "own_alt", "alt_diff", "t",
                 "t_remaining", "dist_to_boundary", "opp_dist_to_boundary")}




# A duel is a task where *both* aircraft can die, and that adds one outcome the
# default `verdict` has no branch for.  It lives here rather than in one of the
# two duels because neither owns it: an even fight is not a kind of uneven one.
MUTUAL_HEALTH_TOL = 0.02
"""Below this, the survivor traded rather than won.

One tick of the 3-hit gun is 0.0165, so without a tolerance a match could be
awarded on a margin thinner than the smallest change the weapon can make.  It
applies to every task, every seed, jitter or no jitter, whether or not the seats
are swapped -- a single engagement decided by less than this was not decided.
"""


class DuelEnv(TaskEnv):
    """Both sides armed, and `mutual` added to the outcomes.

    The starting geometry is what the subclasses differ in, and it is the whole
    difference: `AdvantagedFightEnv` deals the learner the initiative most of
    the time, `FairFightEnv` deals neither side anything.
    """

    channels = (HEADING, SPEED)         # the vertical is closed
    t_max = 120.0
    track_lock = 1.0
    # 0.33 -- three one-second bursts, not one.  The value is a game balance
    # knob and this is where it was settled: at 1.0 a head-on pass kills both
    # aircraft 0.75 of the time, at 0.33 it is 0.37, and `MUTUAL_HEALTH_TOL`
    # below is sized against this gun's 0.0165 per tick.  It lives in the class
    # rather than in a caller because it used to: the answer key set it by
    # subclassing every environment it scored, so the package shipped a
    # different weapon from the one every measurement was taken with.
    flat_damage = 0.33
    wez_cone_deg = 30.0
    foe_armed = True                    # symmetric, and the point of a duel

    def verdict(self, c: Combat):
        """Adds `mutual` to the default's outcomes.  `won` stays a bool.

        A draw is *not* signalled by `won=None`.  A reward function reads `won`
        through `info.get("won")`, which is also None on every step before the
        last one, so a tri-state `won` would make "the match was a draw" and
        "the match is still running" the same value.  Anything that wants the
        distinction reads `info["outcome"]`, which is already in the dict.
        """
        dead_foe = c.health[self.foe_seat] <= 0.0
        dead_own = c.health[self.seat] <= 0.0
        if dead_foe and dead_own:
            return True, False, "mutual"
        if dead_foe:
            if c.health[self.seat] <= MUTUAL_HEALTH_TOL:
                return True, False, "mutual"
            return True, True, "kill"
        if dead_own:
            if c.health[self.foe_seat] <= MUTUAL_HEALTH_TOL:
                return True, False, "mutual"
            return True, False, "died"
        if c.t >= c.t_max:
            return True, False, "timeout"
        return False, None, ""
