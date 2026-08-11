"""`StateSpec v1` -- the observation every task hands out.  37 raw channels.

**This is an instrument panel, not a network input.**  Everything is a raw SI
quantity: metres, metres per second, radians, g, seconds.  Nothing is
normalised, nothing is clipped, no angle is pre-wrapped into sin/cos, and no
relative geometry is computed.  Turning this into something a network can learn
from is the student's job, and it is a graded part of the assignment.

Three properties, and each one is load-bearing.

**Task-invariant.**  The same 37 channels come out of every rung of the ladder.
Channels that are constant in a given task stay in the vector anyway -- in the
circular-target task the altitudes never move and `own_health` never drops --
because a policy trained on rung 1 has to load unchanged into rung 3, and that
is impossible if the input width changes underneath it.  "Work out which inputs
are useless" is part of the exercise.

**Raw.**  Measured with behaviour cloning on 2026-08-11: handing over the
precomputed gun geometry (ATA, aspect, lead) is worth two kills out of thirty
against handing over raw 6-DOF -- 29/30 against 27/30.  The representation is
not what makes this task hard, so there is no reason to spend the teaching
opportunity on it.

**Not normalised.**  This one has a cost and it is deliberate.  `x` arrives as
20,000-odd metres and a network fed that directly will diverge, loudly, on the
first backward pass.  The alternative is worse: an earlier draft normalised
position by 10 km and clipped to [-1, 1], which fails *silently* -- the target
of this task orbits at up to 9.6 km radius (325 kt at 1 deg/s), both aircraft
saturate independently, and two jets a kilometre apart read as the same point.
A loud failure a student can debug beats a quiet one nobody sees.

The derived geometry a reward needs -- range, ATA, aspect, `in_wez` -- is not
here.  It comes out in the `info` dict instead, which keeps observation
engineering and reward engineering as separate exercises.

**Why this is at the top of the package and not under `envs/`.**  It is the
contract a student writes against -- `from aircombat_gym.wvr import obs as O` is in
every submission -- and it is deliberately the same for every environment on the
ladder.  Filing it under one of them would suggest it belongs to that one.
"""
from __future__ import annotations

import math

import numpy as np

STATE_SPEC_VERSION = 1

# --- channel names -----------------------------------------------------------
# Two aircraft, own first, then the match state.  The order is frozen: a policy
# binds to it, and `tests/test_envs.py` checks it against this tuple.

_PER_AIRCRAFT = (
    "x", "y", "h",              # position  [m]  east, north, MSL
    "vx", "vy", "vz",           # velocity  [m/s]  east, north, *up*
    "nx", "ny", "nz",           # load factor [g]  body axes, +1 nz level
    "phi", "theta", "psi",      # attitude  [rad]  bank, pitch, heading
    "p", "q", "r",              # body rates [rad/s]
)

_MATCH = (
    "own_health",               # 1.0 -> 0.0
    "opp_health",
    "own_track_time",           # [s] unbroken seconds of firing solution
    "opp_track_time",           # [s] how long he has held one on me
    "own_in_wez",               # 0.0 / 1.0
    "t_remaining",              # [s]
    "dist_to_boundary",         # [m]
)

STATE_NAMES = (tuple(f"own_{n}" for n in _PER_AIRCRAFT)
               + tuple(f"opp_{n}" for n in _PER_AIRCRAFT)
               + _MATCH)

STATE_DIM = len(STATE_NAMES)                       # 37
_INDEX = {n: i for i, n in enumerate(STATE_NAMES)}


def state_names() -> tuple[str, ...]:
    return STATE_NAMES


def index(name: str) -> int:
    """Channel index by name, for a student who wants three of the thirty-seven."""
    return _INDEX[name]


def make_observation_space():
    """Unbounded on purpose -- see the module docstring."""
    from gymnasium import spaces
    return spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32)


# --- encoding ----------------------------------------------------------------

def _aircraft(s) -> list[float]:
    """One aircraft's 15 channels out of an `AircraftState`.

    Velocity is reconstructed from speed, flight-path angle and ground track
    rather than taken from the backend, because the backend reports body-axis
    components and this vector is in world axes -- the frame the positions are
    in, so that differencing them is meaningful.
    """
    vh = s.v * math.cos(s.gamma)
    return [s.x, s.y, s.h,
            math.sin(s.track) * vh, math.cos(s.track) * vh, s.h_dot,
            s.nx, s.ny, s.nz,
            s.phi, s.theta, s.psi,
            s.p, s.q, s.r]


def encode(own, opp, info: dict) -> np.ndarray:
    """`AircraftState` x2 plus one side's `Combat.observe()` dict -> the vector."""
    v = _aircraft(own) + _aircraft(opp) + [
        info["own_health"], info["opp_health"],
        info["track_time"], info["under_track"],
        1.0 if info["in_wez"] else 0.0,
        info["t_remaining"], info["dist_to_boundary"],
    ]
    return np.asarray(v, dtype=np.float32)


def unpack(obs) -> dict:
    """The vector back as a named dict.  For reading, plotting and debugging.

    Students are expected to write their own transform against `STATE_NAMES`;
    this exists so that printing one frame is not an exercise in counting.
    """
    return {n: float(x) for n, x in zip(STATE_NAMES, np.asarray(obs).ravel())}
