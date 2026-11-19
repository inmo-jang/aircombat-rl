"""The arena: two teams from `config.yaml`, on the starts it gives.

    cfg = load_config()                       # config.yaml next to this file
    blue = load_team(cfg["blue"])             # a policy folder, or a scripted pilot
    env = ArenaEnv(cfg["start"], blue, action_mode="discrete")

Shared by `train.py` and `match.py`.  Nothing in `aircombat_gym` is changed:
`ArenaEnv` is a `DuelEnv` -- the gun, the clock and the four outcomes of
`project_04_fair` -- whose `sample()` returns the starts from the config and
the team from the config as the opponent.

A team is either

    a folder      policy.py + wrappers.py + policy_net.*, loaded exactly as
                  `tools.grade` loads a submission
    a name        a scripted pilot from `aircombat_gym.wvr.baselines`:
                  ace, pursuit, lead, circler, evader
"""
from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import yaml

from aircombat_gym.wvr import actions as A
from aircombat_gym.wvr.baselines import BY_NAME, Bot, Evader
from aircombat_gym.wvr.envs.base import (ARENA_R, MUTUAL_HEALTH_TOL, SIDES,
                                         DuelEnv, Initial)

HERE = pathlib.Path(__file__).resolve().parent

# `tools.policies` loads a submission folder the way the marker does.  It lives
# at the repository root, two levels up.
_ROOT = HERE.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
from tools import policies  # noqa: E402

ALT_FT = 20_000.0               # both aircraft; the vertical is closed
SCRIPTED = tuple(sorted(BY_NAME))


# =============================================================================
# config
# =============================================================================

def load_config(path=None) -> dict:
    """`config.yaml` -> dict, with team entries resolved to folders or names."""
    path = pathlib.Path(path) if path else HERE / "config.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("red", "blue", "start"):
        if key not in cfg:
            raise SystemExit(f"{path}: missing `{key}`")
    for side in SIDES:
        start = cfg["start"].get(side)
        if not start:
            raise SystemExit(f"{path}: missing `start.{side}`")
        for k in ("x", "y", "heading", "speed_kt"):
            if k not in start:
                raise SystemExit(f"{path}: missing `start.{side}.{k}`")
        if math.hypot(start["x"], start["y"]) >= ARENA_R:
            raise SystemExit(f"{path}: start.{side} is outside the "
                             f"{ARENA_R / 1000:.0f} km arena")
    cfg.setdefault("swap_seats", True)
    cfg.setdefault("engagements", 20)
    cfg.setdefault("altitude_band_ft", 0)
    cfg.setdefault("time_limit_s", 120)
    cfg["_dir"] = path.parent
    return cfg


def team_path(cfg: dict, side: str):
    """A team entry -> a scripted pilot's name, or a folder path."""
    entry = str(cfg[side])
    if entry.lower() in BY_NAME:
        return entry.lower()
    p = pathlib.Path(entry)
    return p if p.is_absolute() else (cfg["_dir"] / p).resolve()


# =============================================================================
# teams
# =============================================================================

def decoder(mode: str):
    """An action in `mode` -> the (heading, speed, altitude) deltas.

    The same arithmetic as `TaskEnv.decode`, but bound to the *team's* action
    mode rather than the environment's -- so a discrete policy can meet a
    continuous one.
    """
    if mode == "discrete":
        def decode(a):
            h, s = divmod(int(a), len(A.DELTA_SPEED_KT))     # index = 3*h + s
            return A.DELTA_HEADING_DEG[h], A.DELTA_SPEED_KT[s], 0.0
    else:
        def decode(a):
            a = np.clip(np.asarray(a, dtype=np.float32).ravel(), -1.0, 1.0)
            return (float(a[0]) * A.DELTA_HEADING_DEG[-1],
                    float(a[1]) * A.DELTA_SPEED_KT[-1], 0.0)
    return decode


class Team:
    """One side: what it is called, how it acts, and how it is described."""

    def __init__(self, name: str, act=None, mode: str | None = None,
                 note: str = "", folder: pathlib.Path | None = None):
        self.name, self.act, self.mode = name, act, mode
        self.note, self.folder = note, folder

    @property
    def scripted(self) -> bool:
        return self.act is None

    def pilot(self, env) -> Bot:
        """A fresh opponent for one engagement in `env`."""
        if self.scripted:
            return Evader(env._rng) if self.name == "evader" else BY_NAME[self.name]()
        return _PolicyPilot(self.act, decoder(self.mode))


class _PolicyPilot(Bot):
    """A loaded policy flying the opponent's seat, on that seat's observation."""

    name = "policy"

    def __init__(self, act, decode):
        self._act, self._decode = act, decode

    def act(self, info=None, obs=None):
        return self._decode(self._act(obs))


def load_team(entry) -> Team:
    """A scripted pilot's name, or a folder with a trained policy."""
    if isinstance(entry, str):
        return Team(entry, note=f"scripted `{entry}`")
    folder = pathlib.Path(entry)
    if not folder.is_dir():
        raise SystemExit(f"no such team folder: {folder}  "
                         f"(or a scripted pilot: {', '.join(SCRIPTED)})")
    try:
        weights = policies.default_weights(folder)
        act, note, mode = policies.load(folder, weights)
    except policies.Mismatch as e:
        raise SystemExit(f"cannot load {folder.name}: {e}")
    return Team(folder.name, act, mode, note, folder)


# =============================================================================
# the environment
# =============================================================================

def initial(start: dict, rng) -> Initial:
    """The two starts from the config, randomised as it says.

    Position and heading are jittered per aircraft.  Speed is scaled by one
    factor for both, so neither is handed more energy than the config gave it.
    """
    rz = start.get("randomize") or {}
    dp = float(rz.get("position_m", 0.0))
    dh = float(rz.get("heading_deg", 0.0))
    k = 1.0 + rng.uniform(-1.0, 1.0) * float(rz.get("speed_frac", 0.0))
    xy, psi, v = [], [], []
    for side in SIDES:
        s = start[side]
        xy.append((float(s["x"]) + rng.uniform(-dp, dp),
                   float(s["y"]) + rng.uniform(-dp, dp)))
        psi.append(float(s["heading"]) + rng.uniform(-dh, dh))
        v.append(float(s["speed_kt"]) * k)
    return Initial("arena", tuple(xy), tuple(psi), tuple(v), (ALT_FT, ALT_FT))


class ArenaEnv(DuelEnv):
    """`project_04_fair`'s rules, with the starts and the opponent from config.

    Two ways to put the learner on the other start:

        seat         `reset(options={"seat": "blue"})`, or drawn per reset with
                     `swap_seats` -- the learner flies the blue aircraft.  This
                     is what training uses
        swapped      `reset(options={"swapped": True})` -- the learner keeps
                     its seat, and with it its colour, and the two starts trade
                     places.  This is what a match uses

    On the same seed the two are the same engagement: the start is drawn as
    configured and the two aircraft are then exchanged whole.

    `altitude_band_ft` > 0 adds one rule to the duel's: an aircraft more than
    that far from `ALT_FT` loses, and both at once is `mutual`.  A kill on the
    same step stands.  The outcome is reported as `kill`, `died` or `mutual`,
    so a reward that knows those keeps working; `info["disqualified"]` says
    which seat left the band -- `"red"`, `"blue"` or `"both"`.

    For a match only -- training leaves both off and keeps the duel's 120 s
    `timeout`.  `time_limit_s` sets the clock.  With `decide_on_health`, a
    clock that runs out with both still flying is won by more health --
    reported as `kill` or `died`, with `info["decided_on"] = "health"`.
    Within `MUTUAL_HEALTH_TOL` of each other, less than one tick of the gun,
    it stays a `timeout`.
    """

    def __init__(self, start: dict, opponent: Team, *, swap_seats=False,
                 altitude_band_ft=0.0, time_limit_s=None, decide_on_health=False,
                 action_mode="discrete", seed=None, seat=None):
        self.start, self.opponent, self.swap_seats = start, opponent, swap_seats
        self.altitude_band_ft = float(altitude_band_ft or 0.0)
        self.decide_on_health = bool(decide_on_health)
        if time_limit_s:
            # before the base class builds `Combat`, which takes its clock from here
            self.t_max = float(time_limit_s)
        self.swapped, self._dq, self._on_health = False, None, False
        self._seat_rng = np.random.default_rng(seed)
        super().__init__(action_mode=action_mode, seed=seed, seat=seat)

    def verdict(self, c):
        ended, won, reason = super().verdict(c)
        self._dq, self._on_health = None, False
        if reason == "timeout" and self.decide_on_health:
            lead = c.health[self.seat] - c.health[self.foe_seat]
            if abs(lead) > MUTUAL_HEALTH_TOL:
                self._on_health = True
                return True, lead > 0, ("kill" if lead > 0 else "died")
            return ended, won, reason
        if ended or not self.altitude_band_ft:
            return ended, won, reason
        out = {s: abs(c.ac[s].state.h_ft - ALT_FT) > self.altitude_band_ft
               for s in SIDES}
        if out[self.seat] and out[self.foe_seat]:
            self._dq = "both"
            return True, False, "mutual"
        if out[self.seat]:
            self._dq = self.seat
            return True, False, "died"
        if out[self.foe_seat]:
            self._dq = self.foe_seat
            return True, True, "kill"
        return False, None, ""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._dq:
            info["disqualified"] = self._dq
        if self._on_health:
            info["decided_on"] = "health"
        return obs, reward, terminated, truncated, info

    def sample(self, rng):
        ic = initial(self.start, rng)
        if self.swapped:
            ic = Initial(ic.name, ic.xy[::-1], ic.psi_deg[::-1], ic.v_kt[::-1],
                         ic.alt_ft[::-1])
        return ic, self.opponent.pilot(self)

    def reset(self, *, seed=None, options=None):
        # The seed fixes the start.  The flight that follows can still differ
        # with what this env flew before: a reset does not clear every state.
        self.last_seed = seed
        self.swapped = bool((options or {}).get("swapped", False))
        seat = (options or {}).get("seat")
        if seat is None and self.swap_seats:
            r = (np.random.default_rng(seed) if seed is not None
                 else self._seat_rng)
            seat = SIDES[int(r.integers(len(SIDES)))]
        if seat is not None:
            if seat not in SIDES:
                raise ValueError(f"seat {seat!r} not in {SIDES}")
            self.seat = seat
        return super().reset(seed=seed, options=options)
