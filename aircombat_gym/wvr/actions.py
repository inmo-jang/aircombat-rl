"""The action grid: what one decision can ask for.

    Discrete(27) = heading 3 x speed 3 x altitude 3

Every channel is three-valued -- back off, hold, push -- and a task opens only
the ones it wants (`wvr/envs/base.py`).  **Every task shipped today closes the
vertical, so the only space a student ever sees is `Discrete(9)`.**  27 is the
ceiling, reachable by opening `ALTITUDE`, and nothing does yet.

**Why one flat `Discrete` and not `MultiDiscrete([3, 3])`.**  The factored space
is the better description -- the axes are independent, and a network with one
head per axis learns "turn left" once instead of once per speed setting, because
in the flat space `(-30, +20)` and `(-30, -20)` are unrelated labels under a
single softmax.  Gymnasium has the type for exactly this, and LAG uses it
(`MultiDiscrete([41, 41, 41, 30])` for its four control channels).

It is flattened here for one reason: **Stable-Baselines3's DQN accepts
`Discrete` and nothing else.**  PPO and A2C take `MultiDiscrete`; DQN does not,
and DQN is the algorithm this environment is taught with, so a factored space
would fail on the first line a student writes.  Flattening is also ordinary --
Atari's `Discrete(18)` is 9 joystick directions times fire.

The cost is a product rather than a sum, and it is affordable at 9 and at 27.
If the vertical opens *and* a weapon channel arrives, 81 or 162 is the point to
revisit this -- by then the course would have to be PPO-only anyway.

The observation spec is not here: it is `wvr/obs.py`, because it is the same for
every task while the open axes are not.

Two configurations were dropped on the way here, both from the 2D version:

  the locked pair pinned altitude to H0.  With the vertical shut, energy
  management does not pay -- the sustained turn converges to one circle whatever
  the entry speed.  Opening it puts the trade back: peak turn rate is 17.7 deg/s
  at 5,000 ft against 9.4 at 30,000.  That is the game.

  the +-15 deg heading steps, which made the channel five wide.  A scripted lead
  pursuit is 2.7x slower to kill without them, because the smallest turn it can
  ask for is wider than the firing cone and its aim never settles.  That is a
  lower bound and does not decide the question: a script rounds its demand to the
  nearest grid point and so cannot do what a learned policy can -- alternate +30
  and 0 across steps to average something in between.  Measured on a learned
  policy the finer grid bought nothing, and the narrower space is cheaper to
  explore: a uniform random policy meets each of 27 actions half again as often
  as each of 45.
"""
from __future__ import annotations

# Deltas, re-applied every decision step: target = current + delta, except that
# a zero delta *freezes* the target rather than re-deriving it (see aircraft.py).
# Holding one action is therefore a sustained maximum-performance manoeuvre and
# the zero action means "keep what you have".
DELTA_HEADING_DEG = (-30.0, 0.0, 30.0)
DELTA_SPEED_KT = (-20.0, 0.0, 20.0)
DELTA_ALT_FT = (-1000.0, 0.0, 1000.0)

N_HEADING = len(DELTA_HEADING_DEG)
N_SPEED = len(DELTA_SPEED_KT)
N_ALT = len(DELTA_ALT_FT)

DISCRETE_N = N_HEADING * N_SPEED * N_ALT       # 27, all three channels open


def action_n() -> int:
    """The full three-channel grid.  An assignment that closes a channel builds
    its own space; this is the ceiling."""
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
