"""The two aircraft, the weapon bookkeeping, and the Gymnasium surface.

Two classes, and the split between them is the one that matters:

    Combat    runs both aircraft and *records* what the weapon did -- health,
              how long each side has held a firing solution, the clock.  It
              decides nothing.
    TaskEnv   the Gymnasium environment.  It decides: `verdict()` says what ends
              a match and who won, and each assignment overrides it as far as it
              needs to.

Judging used to happen in both places -- a `Report` built here and then ignored
by the env, which re-derived the same answer from health and time.  Two judges
drift; the timeout rule already had.

**No reward.**  `step` returns 0.0, always.  The material for a reward goes out
in `info` and the student writes the function -- `tests/test_envs.py` walks the
package AST to keep it that way.

An assignment is one file next to this one.  It supplies `sample()`, sets a few
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
ARENA_R = 10_000.0

# Room enough that nothing bumps into it.  A task with a real arena passes its
# own radius; one without says so with a number rather than with a special case,
# so `dist_to_boundary` stays a finite metre count in every task.
NO_ARENA_M = 1.0e6

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
                 arena_m: float = ARENA_R) -> None:
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
        """
        self.t_max = t_max
        self.track_lock = track_lock
        self.flat_damage = flat_damage
        self.wez_cone_deg = wez_cone_deg
        self.arena_m = arena_m
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
            self.ac[s].guidance.h0_ft = ic.alt_ft[k]
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
                damage_rate=e.damage_rate,
                own_speed=st[s].v_kt, opp_speed=st[o].v_kt,
                own_health=self.health[s], opp_health=self.health[o],
                dist_to_boundary=min(self.arena_m,
                                     ARENA_R - math.hypot(st[s].x, st[s].y)),
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
        """
        for s in SIDES:
            self.ac[s].step(*actions[s])
        dt = 1.0 / _ac.DECISION_HZ
        self.t += dt

        info = self.observe()
        for s in SIDES:
            o = "blue" if s == "red" else "red"
            e = self._eng[s]
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
        self.min_range = min(self.min_range, self._eng["red"].r)
        return info

    # --- facts a verdict may want.  Still not a judgement. -------------------

    def out_of_bounds(self, side: str) -> bool:
        st = self.ac[side].state
        return (math.hypot(st.x, st.y) > ARENA_R
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
    """One learner (red) against one scripted opponent (blue).

    An assignment subclasses this, implements `sample()`, and sets whichever
    class attributes differ.  `verdict()` is where its difficulty lives.
    """

    metadata = {"render_modes": []}

    # --- what an assignment configures ---
    channels: tuple[str, ...] = (HEADING, SPEED)
    t_max: float = 120.0
    track_lock: float = 1.0
    flat_damage: float | None = 1.0
    wez_cone_deg: float = 30.0
    arena_m: float = NO_ARENA_M
    # The rest of the sight.  Class attributes so an assignment can narrow the
    # band, and so a viewer can read the numbers in force off the env instead of
    # importing the weapon module and hoping they match.
    wez_r_min: float = WEZ_R_MIN
    wez_r_max: float = WEZ_R_MAX
    muzzle_ms: float = MUZZLE_MS

    def __init__(self, action_mode: str = "discrete", seed: int | None = None):
        super().__init__()
        if action_mode not in ("discrete", "continuous"):
            raise ValueError(f"action_mode {action_mode!r}")
        self.action_mode = action_mode
        self._grid = [_GRID[c] for c in _ORDER if c in self.channels]
        self._slot = [_ORDER.index(c) for c in _ORDER if c in self.channels]
        self._scale = np.array([abs(g[-1]) for g in self._grid], dtype=np.float32)

        if action_mode == "discrete":
            n = 1
            for g in self._grid:
                n *= len(g)
            self.action_space = gym.spaces.Discrete(n)
        else:
            self.action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(len(self._grid),), dtype=np.float32)
        self.observation_space = O.make_observation_space()

        self._combat = Combat(t_max=self.t_max, track_lock=self.track_lock,
                              flat_damage=self.flat_damage,
                              wez_cone_deg=self.wez_cone_deg,
                              arena_m=self.arena_m)
        self._rng = np.random.default_rng(seed)

    # --- what an assignment supplies -----------------------------------------

    def sample(self, rng):
        """Draw one episode: `(Initial, opponent)`.  The opponent is pre-reset."""
        raise NotImplementedError

    def verdict(self, c: Combat):
        """`(ended, won, reason)`.  Override to change an assignment's rules.

        The default is the simplest thing that is a game: someone dies, or the
        clock runs out and the learner has not won.  Leaving the arena is *not*
        a loss here -- assignment 01 has no arena, and a task that does gives
        one by overriding this and calling `super()`.
        """
        if c.health["blue"] <= 0.0:
            return True, True, "kill"
        if c.health["red"] <= 0.0:
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
        return self._obs(info), self._info(info)

    def step(self, action):
        # the opponent decides from the state it can see *now*, before the step
        foe_action = self._foe.act(self._combat.observe()["blue"])
        info = self._combat.step({"red": self.decode(action),
                                  "blue": foe_action})
        ended, won, reason = self.verdict(self._combat)

        extra = self._info(info)
        if ended:
            extra.update(outcome=reason, won=won, t=self._combat.t,
                         wez_time=self._combat.wez_time["red"])
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

    def _obs(self, info: dict) -> np.ndarray:
        return O.encode(self._combat.ac["red"].state,
                        self._combat.ac["blue"].state, info["red"])

    def _info(self, info: dict) -> dict:
        """The material a reward is built from.  Derived geometry lives here,
        never in the observation -- see `wvr/obs.py`."""
        i = info["red"]
        return {k: i[k] for k in
                ("range", "range_rate", "ata", "ata_signed", "ata_lead",
                 "lead_signed", "aa", "in_wez", "track_time", "under_track",
                 "damage_rate", "own_health", "opp_health", "own_speed",
                 "opp_speed", "own_alt", "alt_diff", "t", "t_remaining",
                 "dist_to_boundary")}

