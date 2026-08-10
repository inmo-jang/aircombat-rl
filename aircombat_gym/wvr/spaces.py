"""Frozen observation and action spec.

    obs 18, Discrete(27) = heading 3 x speed 3 x altitude 3.

Every channel is three-valued: back off, hold, push.  DQN only supports
Discrete, which is why they are flattened rather than left as MultiDiscrete.

Two configurations were dropped on the way here, both from the 2D version:

  the locked obs/action pair (obs 14, Discrete(15)) pinned altitude to H0.  With
  the vertical shut, energy management does not pay -- the sustained turn
  converges to one circle whatever the entry speed, so energy buys only a
  transient (workplan 3.2.9, 3.2.11).  Opening it puts the trade back: peak turn
  rate is 17.7 deg/s at 5,000 ft against 9.4 at 30,000.  That is the game.

  the +-15 deg heading steps, which made the channel five wide.  A scripted lead
  pursuit is 2.7x slower to kill without them (91 s -> 251 s over six starting
  geometries) because the smallest turn it can then ask for is wider than the
  15 deg firing cone, so its aim never settles.  That measurement is a lower
  bound and does not decide the question: the script rounds its demand to the
  nearest grid point and therefore cannot do the thing a learned policy can --
  alternate +30 and -30 across steps to hold a heading between them.  Whether
  DQN finds that is an empirical question to settle with DQN, and the narrower
  space is worth the try: a uniform random policy meets each of 27 actions half
  again as often as each of 45.  Revisit if training stalls at the aiming stage.
"""
from __future__ import annotations

import numpy as np

# Covers the action space as well as the observation -- a policy binds to both,
# and `test_spec_is_frozen` checks both against this number.
#   v4  obs 18, Discrete(45)   heading 5 wide
#   v5  obs 18, Discrete(27)   heading 3 wide
OBS_SPEC_VERSION = 5

# --- action increments -------------------------------------------------------
# Deltas, re-applied every decision step: target = current + delta, except that
# a zero delta *freezes* the target rather than re-deriving it (see aircraft.py).
# Holding one action is therefore a sustained maximum-performance manoeuvre and
# the noop action means "keep what you have".
DELTA_HEADING_DEG = (-30.0, 0.0, 30.0)
DELTA_SPEED_KT = (-20.0, 0.0, 20.0)
DELTA_ALT_FT = (-1000.0, 0.0, 1000.0)

N_HEADING = len(DELTA_HEADING_DEG)
N_SPEED = len(DELTA_SPEED_KT)
N_ALT = len(DELTA_ALT_FT)

DISCRETE_N = N_HEADING * N_SPEED * N_ALT       # 27

# --- observation -------------------------------------------------------------
# `energy_height` is the quantity that actually matters -- h + V^2/2g, what the
# aircraft could climb to if it spent all its speed -- and it is not recoverable
# from altitude and speed separately once the policy can trade between them.
OBS_NAMES = (
    "own_speed",          # V / V_max
    "own_health",
    "range",
    "ata_sin", "ata_cos",
    "aa_sin", "aa_cos",
    "range_rate",
    "opp_speed",
    "opp_health",
    "dist_to_boundary",
    "bearing_center_sin", "bearing_center_cos",
    "t_remaining",
    "own_alt",            # (h - h_mid) / h_half
    "alt_diff",           # (h_opp - h) / h_half -- who is on top
    "own_energy_height",
    "opp_energy_height",
)

OBS_DIM = len(OBS_NAMES)                       # 18

# normalisation references
R_REF_M = 1500.0
V_MAX_REF_KT = 650.0
ALT_MID_FT = 17500.0
ALT_HALF_FT = 12500.0
EH_REF_FT = 45000.0


def obs_dim() -> int:
    return OBS_DIM


def obs_names() -> tuple[str, ...]:
    return OBS_NAMES


def action_n() -> int:
    return DISCRETE_N


def decode(a) -> tuple[float, float, float]:
    """Action index -> (delta_heading_deg, delta_speed_kt, delta_alt_ft)."""
    a = int(a)
    if not 0 <= a < DISCRETE_N:
        raise ValueError(f"action {a} outside Discrete({DISCRETE_N})")
    i_h, rem = divmod(a, N_SPEED * N_ALT)
    i_v, i_a = divmod(rem, N_ALT)
    return DELTA_HEADING_DEG[i_h], DELTA_SPEED_KT[i_v], DELTA_ALT_FT[i_a]


def noop() -> int:
    """The index that means "hold everything" -- a fresh policy should find it."""
    return ((DELTA_HEADING_DEG.index(0.0) * N_SPEED + DELTA_SPEED_KT.index(0.0))
            * N_ALT + DELTA_ALT_FT.index(0.0))


def make_action_space():
    from gymnasium import spaces
    return spaces.Discrete(DISCRETE_N)


def make_observation_space():
    from gymnasium import spaces
    return spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
