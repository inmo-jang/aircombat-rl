"""Regression tests for the flight layer.

Every test here exists because something silently went wrong once.  Four are the
ones docs/conversation_log.md section 10 marks as "must be reproduced":
position signs, crossing the origin, the gun line at the vertical, and reset
repeatability after violent flight.  None of the four raised an exception when
they were broken -- all showed up as "this number looks wrong", which is exactly
what a test suite is for.

The game and learning layers are in `archived/` for now, and their tests went
with them (`archived/test_game_and_learning.py`).  Bring the tests
back with the module they cover -- restoring code without its test is how this
project lost a day to four mistakes in one session.

These live outside the package on purpose.  `aircombat_gym` is what students
install and import; a test suite is not part of what it does, and keeping it out
means the tests exercise the package the same way a student does -- through the
installed import, not through a relative one.

    python -m pytest -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aircombat_gym.wvr import actions as act
from aircombat_gym.core.aircraft import Aircraft
from aircombat_gym.core.envelope import (H0_FT, N_STRUCT, level_turn_rate_deg_s,
                                         max_bank_deg, n_max)


def test_spec_is_frozen():
    assert act.action_n() == 27
    assert act.DELTA_HEADING_DEG == (-30.0, 0.0, 30.0)
    assert act.DELTA_SPEED_KT == (-20.0, 0.0, 20.0)
    assert act.DELTA_ALT_FT == (-1000.0, 0.0, 1000.0)
    assert act.DELTA_HEADING_DEG == (-30.0, 0.0, 30.0)


def test_noop_action_holds_everything():
    """A freshly initialised policy must be able to fly straight and level."""
    assert act.decode(act.noop()) == (0.0, 0.0, 0.0)


def test_every_action_decodes_and_is_unique():
    seen = {act.decode(a) for a in range(act.action_n())}
    assert len(seen) == act.action_n()


# --------------------------------------------------------------------------
# gate H5 -- simulator hygiene.  The engine being off silently invalidated a
# whole set of measurements once (workplan 3.2.4).
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kt", [300.0, 450.0, 600.0])
def test_h5_trimmed_flight_is_sane(kt):
    ac = Aircraft()
    s = ac.reset(v_kt=kt)
    assert s.thrust > 0.0, "engine is not running"
    assert 0.90 <= s.nz <= 1.10, f"trim is not 1 g ({s.nz:.2f})"
    for _ in range(60):
        s = ac.step(0.0, 0.0, 0.0)
    assert abs(s.h_ft - H0_FT) < 50.0
    assert abs(math.degrees(s.gamma)) < 2.0


# --------------------------------------------------------------------------
# the four the conversation log marks as mandatory
# --------------------------------------------------------------------------

def test_position_signs_follow_the_heading():
    """North-east-south-west must move x and y the way the compass says."""
    for psi_deg, dx, dy in ((0, 0, +1), (90, +1, 0), (180, 0, -1), (270, -1, 0)):
        ac = Aircraft()
        a = ac.reset(psi=math.radians(psi_deg), v_kt=450.0)
        for _ in range(100):
            b = ac.step(0.0, 0.0, 0.0)
        if dx:
            assert math.copysign(1, b.x - a.x) == dx, f"x wrong at {psi_deg} deg"
        if dy:
            assert math.copysign(1, b.y - a.y) == dy, f"y wrong at {psi_deg} deg"


def test_crossing_the_origin_is_monotonic():
    """position/distance-from-start-* is unsigned and reflects at the origin.

    Flying south from north of it must decrease y the whole way, with no jump.
    """
    ac = Aircraft()
    ac.reset(x=0.0, y=3000.0, psi=math.pi, v_kt=450.0)
    ys = [ac.step(0.0, 0.0, 0.0).y for _ in range(600)]
    diffs = np.diff(ys)
    assert (diffs < 0).all(), "y did not decrease monotonically"
    assert abs(diffs).max() < 3.0 * abs(diffs).mean(), "discontinuity crossing y=0"
    assert ys[-1] < 0.0, "never actually crossed the origin"


def test_gun_line_does_not_spin_at_the_vertical():
    """Euler angles are singular near vertical; the nose vector must not jump."""
    ac = Aircraft()
    ac.reset(v_kt=500.0)
    prev = None
    worst = 0.0
    for i in range(400):
        s = ac.step(30.0, 0.0, 1000.0 if i < 200 else -1000.0)
        nx, ny = s.nose_2d
        if prev is not None and s.nose_valid and prev[2]:
            d = math.degrees(math.acos(max(-1.0, min(1.0,
                             nx * prev[0] + ny * prev[1]))))
            worst = max(worst, d)
        prev = (nx, ny, s.nose_valid)
    assert worst < 20.0, f"gun line jumped {worst:.0f} deg in one decision step"


def test_reset_is_repeatable_after_violent_flights():
    """run_ic() alone leaves the FCS integrators loaded (TRAP-2).

    This is the test that would have saved an entire H4 tournament.
    """
    def probe(ac):
        ac.reset(v_kt=450.0)
        for _ in range(60):
            ac.step(30.0, 0.0, 0.0)
        s = ac.state
        return np.array([s.h_ft, s.psi, s.v_kt, s.nz, s.phi])

    fresh = probe(Aircraft())
    used = Aircraft()
    probe(used)
    for _ in range(400):
        used.step(30.0, -20.0, -1000.0)      # something violent
    after = probe(used)
    assert np.abs(fresh - after).max() < 1e-6, (
        f"reset left state behind: {np.abs(fresh - after)}")


# --------------------------------------------------------------------------
# envelope and the 2D lock
# --------------------------------------------------------------------------

def test_turn_rate_peaks_at_corner_speed():
    rates = {v: level_turn_rate_deg_s(float(v), H0_FT) for v in range(300, 651, 50)}
    best = max(rates, key=rates.get)
    assert 450 <= best <= 550, f"corner speed drifted to {best} kt"


def test_slower_means_less_bank_available():
    """max_bank(V) is what stops "fly slowest" from being a dominant strategy."""
    assert max_bank_deg(300.0) < max_bank_deg(400.0) < max_bank_deg(500.0)


def test_altitude_buys_turn_performance():
    """The whole reason the vertical is worth opening."""
    low = level_turn_rate_deg_s(500.0, 10000.0)
    high = level_turn_rate_deg_s(500.0, 30000.0)
    assert low > 1.5 * high, f"altitude barely matters: {low:.1f} vs {high:.1f}"


def test_structural_limit_is_enforced():
    assert n_max(700.0, 5000.0) <= N_STRUCT + 1e-9


def test_altitude_lock_holds_under_random_commands():
    """Gate L2, shortened.  The full 300 s version lives in preflight/."""
    rng = np.random.default_rng(0)
    ac = Aircraft()
    ac.reset(v_kt=400.0)
    worst = 0.0
    for _ in range(1200):                      # 60 s
        s = ac.step(float(rng.choice([-30, -15, 0, 15, 30])),
                    float(rng.choice([-20, 0, 20])), 0.0)
        worst = max(worst, abs(s.h_ft - H0_FT))
    assert worst < 150.0, f"lock leaked {worst:.0f} ft"


def test_a_tap_climbs_by_the_amount_it_asked_for():
    """The vertical channel has to actually go where it was sent.

    This replaces an older test that asserted the opposite -- that no action
    could leave H0 -- which was the 2D-era `guidance_locked` configuration.  That
    configuration is gone: with the vertical shut, energy management does not pay
    (workplan 3.2.9), so opening it is the game rather than an option.  What
    needs guarding now is that the channel works in both directions and that a
    zero delta still freezes the target it arrived at.
    """
    for delta in (1000.0, -1000.0):
        ac = Aircraft()
        ac.reset(v_kt=450.0)
        h0 = ac.state.h_ft
        ac.step(0.0, 0.0, delta)          # one tap
        for _ in range(600):              # then hold: target must stay put
            s = ac.step(0.0, 0.0, 0.0)
        climbed = s.h_ft - h0
        assert abs(climbed - delta) < 100.0, (
            f"tap of {delta:+.0f} ft moved {climbed:+.0f} ft")


# --------------------------------------------------------------------------
# action semantics
# --------------------------------------------------------------------------


def test_a_tap_turns_by_the_amount_it_asked_for():
    """One heading action, then hold -- the aircraft must arrive where it said.

    A tap of +30 has to turn 30 deg and stop.  That is the property the whole
    action space rests on, and it is the half of the tap/hold pair a policy uses
    for placement rather than rate: hold the same key and the target is re-aimed
    every step so the turn never ends, tap it once and it lands.

    It also guards the freeze semantics.  If a zero delta stopped freezing the
    target and started re-deriving it from the current state, the aircraft would
    drift past instead of arriving, and there would be no way to stop turning.

    There used to be a stronger claim here -- that +-15 and +-30 must also differ
    in *sustained* rate while held -- which forced the heading law into a taper.
    That was a 2D-era requirement (gate H3) and it cost 12-31 % of the settling
    time across the envelope, so both the requirement and the +-15 actions are
    gone.  Under the proportional law any command past about 4 deg of error
    saturates the same bank ceiling.
    """
    for delta in (-30.0, 30.0):
        ac = Aircraft()
        ac.reset(v_kt=450.0)
        for _ in range(40):
            ac.step(0.0, 0.0, 0.0)
        psi0 = ac.state.psi
        ac.step(delta, 0.0, 0.0)          # one tap
        for _ in range(400):              # then hold: target must stay put
            s = ac.step(0.0, 0.0, 0.0)
        turned = math.degrees((s.psi - psi0 + math.pi) % (2 * math.pi) - math.pi)
        assert abs(turned - delta) < 3.0, (
            f"tap of {delta:.0f} deg turned {turned:.1f} deg")


# --------------------------------------------------------------------------
# reward -- the rule the students are taught, applied to our own example
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------

def test_the_environment_does_not_import_a_learning_library():
    """`aircombat_gym` is the gym.  Training and grading consume it, never the
    way round.

    The rule is easy to state and easy to break by accident -- one convenience
    import of a training helper and the package stops being installable without
    torch, which is exactly what `pip install aircombat-gym` promises.  The same
    rule kept the package free of autonomy_bt through the whole time it lived
    inside that repo, and it is what made moving out a directory copy.
    """
    import ast
    import pathlib

    import aircombat_gym

    banned = {"torch", "stable_baselines3", "sb3_contrib", "training", "grading"}
    root = pathlib.Path(aircombat_gym.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for n in names:
                if n in banned:
                    offenders.append(f"{path.relative_to(root)}: {n}")
    assert not offenders, "the gym reached into the training side: " + str(offenders)
