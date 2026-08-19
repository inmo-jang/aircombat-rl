"""Flight envelope constants, measured from JSBSim F-16 at H0.

Every number was measured at H0, not guessed.  Change H0 and they have to be
measured again: `n_max` is a function of dynamic pressure.

The one that matters most is `max_bank`.  It is what makes the 2D lock possible:
by never commanding more bank than the current speed can hold in level flight,
the altitude loop is never asked for lift the aircraft cannot make.  As a side
effect it also produces corner speed as an emergent rule -- slow aircraft simply
cannot turn.
"""
from __future__ import annotations

import bisect
import math

# --- altitude ----------------------------------------------------------------
# H0 is where episodes start.  Whether the policy may leave it is a config
# choice: locked (v3) or open (v4).  Opening it is what puts energy management
# back in the game -- peak turn rate runs 17.7 deg/s at 5,000 ft against
# 9.4 at 30,000, so altitude buys turn performance and not just speed.
H0_FT = 20000.0
ALT_MIN_FT = 5000.0
ALT_MAX_FT = 30000.0

# --- where the arena sits on the globe --------------------------------------
# Only TacView cares.  The game runs in a local tangent plane whose origin is
# this point, so moving it changes nothing physical.  Seoul, so replays open
# somewhere recognisable.
#
# NOTE: set the initial condition through ic/lat-geod-deg, not ic/lat-gc-deg.
# The latter is geocentric and lands the aircraft 0.19 deg (21 km) north.
CENTER_LAT_DEG = 37.5665
CENTER_LON_DEG = 126.9780

# --- speed band (D24) -------------------------------------------------------
# 150 kt measures n_max = 0.90, so level flight is impossible there, but the
# real floor is much higher than "can it fly".  Under 300 s of adversarial 20 Hz
# random commands the altitude lock holds to 93 ft at 300 kt, 157 ft at 275 and
# 342 ft at 250: below 300 the aircraft is lift-starved, the elevator saturates
# a quarter of the time and it can no longer climb back to H0 while turning.
#
# The band still carries the game: turn rate runs 8.4 deg/s at the floor,
# 12.9 at corner speed and 10.3 at the ceiling -- a 1.53x spread with a clear peak.
#
# The ceiling was set to what the aircraft could reach in level flight at the
# old 0.75 throttle cap -- 649 KTAS, Mach 1.11 here -- because commanding more
# than it can reach just creates a dead zone at the top of the action range.
# With the cap lifted it is reachable with margin (0.90 would take it to 699),
# and it is now the binding constraint: the speed loop targets 650 and never
# asks for more throttle than that needs, about 0.89.
V_MIN_KT = 300.0
V_MAX_KT = 650.0

# --- throttle ceiling (D23) -------------------------------------------------
# A game-balance knob, not physics.  Uncapped: `V_MAX_KT` binds above ~0.89, so
# raising it past 0.9 changes nothing in level flight -- what it changes is how
# fast the aircraft recovers speed.
THROTTLE_CAP = 1.00

# --- lift held back for altitude control -------------------------------------
# An *absolute* reserve, not a percentage.  0.9 * n_max leaves 0.6 g spare at
# corner speed but only 0.25 g at the floor, which is where the aircraft
# actually needs it -- and that is what produced a 192 ft excursion before this
# was changed.  Autopilot allocates the same way: altitude takes its g first and
# the bank ceiling gets what is left (see control/autopilot.py).
N_RESERVE = 0.35

# --- structural limit --------------------------------------------------------
# The bundled f16 has no g limiter of its own -- measured n_max keeps climbing
# past 9 at high speed, which would let a fast aircraft turn nearly as well as
# one at corner speed and flatten the whole energy game.  The real F-16 is a
# 9 g airframe.
N_STRUCT = 9.0

G = 9.80665
KT_TO_MS = 0.514444

# --- measured maximum load factor vs speed AND altitude, full AB -------------
# Peak Nz within 1.5 s of full aft stick from trimmed level flight, capped at
# N_STRUCT.  Below roughly 450 kt the alpha limiter binds; above it the FLCS
# g-command saturates, except low down where the 9 g airframe limit is real.
#
# The altitude axis is what makes the vertical worth flying.  Peak turn rate:
#    5,000 ft  17.7 deg/s @ 500 kt
#   20,000 ft  12.8 deg/s @ 500 kt
#   30,000 ft   9.4 deg/s @ 450 kt
# Almost a factor of two.  Climbing stores energy but costs turn performance
# while you are up there; diving buys both speed and a better wing.  That trade
# is the thing a locked-altitude game cannot have.
#
# CAVEAT: these are 1.5 s *transient* peaks.  Below about 300 kt the aircraft
# cannot hold them -- in sustained flight alpha climbs past 20 deg and it
# delivers barely 1 g.  Hence V_MIN_KT = 300 and the absolute N_RESERVE.
_SPD_PTS = [250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
_ALT_PTS = [5000.0, 10000.0, 15000.0, 20000.0, 25000.0, 30000.0]
N_MAX_TABLE: dict[float, list[float]] = {
    5000.0:  [3.16, 4.08, 5.04, 6.09, 7.26, 8.50, 9.00, 9.00, 9.00, 9.00],
    10000.0: [2.81, 3.66, 4.56, 5.56, 6.68, 7.86, 8.05, 8.04, 8.22, 9.00],
    15000.0: [2.47, 3.26, 4.09, 5.04, 6.09, 7.20, 7.05, 6.96, 7.37, 8.45],
    20000.0: [2.14, 2.86, 3.63, 4.52, 5.50, 6.30, 6.10, 5.92, 6.55, 7.54],
    25000.0: [1.83, 2.48, 3.18, 4.01, 4.92, 5.40, 5.14, 4.96, 5.78, 6.66],
    30000.0: [1.56, 2.12, 2.75, 3.50, 4.33, 4.55, 4.25, 4.29, 5.06, 5.82],
}


# Where the table stops.  `n_max` clamps to these rather than extrapolating, so
# outside them it returns the edge value and stops responding -- fine for the
# game, whose band is 5,000-30,000 ft, but a flight-test tool that lets you
# leave the band has to say so instead of showing a frozen number.
TABLE_ALT_RANGE_FT = (_ALT_PTS[0], _ALT_PTS[-1])
TABLE_SPEED_RANGE_KT = (_SPD_PTS[0], _SPD_PTS[-1])


def in_measured_table(v_kt: float, h_ft: float) -> bool:
    return (TABLE_ALT_RANGE_FT[0] <= h_ft <= TABLE_ALT_RANGE_FT[1]
            and TABLE_SPEED_RANGE_KT[0] <= v_kt <= TABLE_SPEED_RANGE_KT[1])


def _interp(xs: list[float], ys: list[float], x: float) -> float:
    x = min(max(x, xs[0]), xs[-1])
    i = bisect.bisect_left(xs, x)
    if i == 0:
        return ys[0]
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + (ys[i] - ys[i - 1]) * t


def n_max(v_kt: float, h_ft: float = H0_FT) -> float:
    """Maximum attainable load factor, bilinear in true airspeed and altitude."""
    h = min(max(h_ft, _ALT_PTS[0]), _ALT_PTS[-1])
    j = bisect.bisect_left(_ALT_PTS, h)
    if j == 0:
        return min(N_STRUCT, _interp(_SPD_PTS, N_MAX_TABLE[_ALT_PTS[0]], v_kt))
    h0, h1 = _ALT_PTS[j - 1], _ALT_PTS[j]
    lo = _interp(_SPD_PTS, N_MAX_TABLE[h0], v_kt)
    hi = _interp(_SPD_PTS, N_MAX_TABLE[h1], v_kt)
    return min(N_STRUCT, lo + (hi - lo) * (h - h0) / (h1 - h0))


def n_usable(v_kt: float, h_ft: float = H0_FT) -> float:
    """Load factor autopilot is allowed to plan on, reserve already removed."""
    return max(1.05, n_max(v_kt, h_ft) - N_RESERVE)


def max_bank_rad(v_kt: float, h_ft: float = H0_FT) -> float:
    """Largest bank angle that can be held in level flight at this speed.

    A level turn needs n = 1/cos(phi).  Invert that against the lift the
    aircraft can actually make, hold the reserve back, and you get a bank
    ceiling that shrinks as the aircraft bleeds speed.

    Autopilot recomputes this every step with the altitude correction subtracted
    first, so this is the ceiling with nothing else competing -- what the HUD
    shows and what the bots reason about.
    """
    n = n_usable(v_kt, h_ft)
    if n <= 1.02:
        return 0.0
    return math.acos(1.0 / n)


def max_bank_deg(v_kt: float, h_ft: float = H0_FT) -> float:
    return math.degrees(max_bank_rad(v_kt, h_ft))


def level_turn_rate_deg_s(v_kt: float, h_ft: float = H0_FT) -> float:
    """Turn rate at max_bank(V) -- what the aircraft can actually deliver now.

    Peaks at corner speed (500 kt, 13.6 deg/s at 20,000 ft): below it the lift
    limit dominates and rate rises with speed, above it the g ceiling is fixed
    and rate falls as 1/V.
    """
    n = n_usable(v_kt, h_ft)
    if n <= 1.02:
        return 0.0
    v_ms = v_kt * KT_TO_MS
    return math.degrees(G * math.sqrt(n * n - 1.0) / v_ms)
