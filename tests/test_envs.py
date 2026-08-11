"""The observation spec and the assignments: `wvr/obs.py` and `aircombat_gym/wvr/envs/`.

Four groups, and the last one is the one that matters:

    spec        37 channels, frozen order, raw SI -- a policy binds to all three
    assignment  registration, closed channels, action round-trip
    no reward   D10, enforced by an AST walk over the package
    playable    `ace` can actually win, a policy is expressible from the raw
                observation, nothing goes non-finite

The weapon model and the scripted bots are in `test_wvr.py`; the flight model
and the envelope are in `test_aircombat.py`.

Two of these replace tests that were deleted on 2026-08-11 when `training/` was
archived and the reward moved out of the repo.  The rules they enforced are
still the right rules, so they come back here, checked against the environment instead
of against a module constant:

  * `test_ace_can_actually_win` -- a terminal nobody can reach is not a
    terminal.  The old version did arithmetic on `DAMAGE_RATE`; this one flies.
  * `test_the_package_computes_no_reward` -- D10, now with an AST walk so it cannot
    rot the way a hand-written assertion would.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import aircombat_gym.wvr.envs                                    # registers ids
from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.envs import ENVS
from aircombat_gym.wvr.envs.base import ALTITUDE, HEADING, SPEED
from aircombat_gym.wvr.envs.circular import ALT_FT, CircularTargetEnv
from aircombat_gym.wvr.baselines import Ace, Pursuit

PKG = Path(aircombat_gym.wvr.envs.__file__).resolve().parent.parent


def _env(**kw):
    e = CircularTargetEnv(**kw)
    e.reset(seed=0)
    return e


# --------------------------------------------------------------------------
# the observation spec
# --------------------------------------------------------------------------

def test_spec_is_frozen():
    """A policy binds to the width and the order.  Changing either silently
    turns every saved checkpoint into garbage that still loads."""
    assert O.STATE_SPEC_VERSION == 1
    assert O.STATE_DIM == 37
    assert O.STATE_NAMES[:3] == ("own_x", "own_y", "own_h")
    assert O.STATE_NAMES[15:18] == ("opp_x", "opp_y", "opp_h")
    assert O.STATE_NAMES[-1] == "dist_to_boundary"
    assert len(set(O.STATE_NAMES)) == O.STATE_DIM, "duplicate channel name"


def test_every_task_hands_out_the_same_observation():
    """The point of the spec: rung 1's policy has to load into rung 3."""
    for env_id in ENVS:
        e = gym.make(env_id)
        assert e.observation_space.shape == (O.STATE_DIM,), env_id
        o, _ = e.reset(seed=0)
        assert o.shape == (O.STATE_DIM,), env_id


def test_the_observation_is_raw_si_not_normalised():
    """`obs.py` is an instrument panel.  If someone 'helpfully' normalises it,
    every worked example in the student README stops being true -- and the
    assignment loses the exercise it is built around."""
    d = O.unpack(_env()._obs(_env()._combat.observe()))
    assert abs(d["own_h"] - ALT_FT * 0.3048) < 1.0, "altitude is not in metres"
    assert 150.0 < math.hypot(d["own_vx"], d["own_vy"]) < 320.0, "speed not m/s"
    assert abs(d["own_psi"]) <= math.pi + 1e-6, "heading is not radians"
    assert 0.5 < d["own_nz"] < 1.5, "nz is not g, or not ~1 in level flight"
    # and unbounded, which is what makes the raw values survivable
    lo, hi = _env().observation_space.low, _env().observation_space.high
    assert np.all(np.isinf(lo)) and np.all(np.isinf(hi))


def test_position_channels_would_have_overflowed_a_normalised_box():
    """Why the box is unbounded, kept as a number rather than a comment.

    The target orbits at V/omega; at 325 kt and the easy end of the turn-rate
    range that is 9.6 km, which is most of the +-10 km a normalised position
    channel would have had -- and both aircraft saturate independently, so two
    jets a kilometre apart read as the same point.
    """
    v = 325.0 * 0.514444
    radius = v / math.radians(1.0)
    assert radius > 9_000.0


def test_unpack_round_trips():
    e = _env()
    o = e._obs(e._combat.observe())
    assert np.allclose(list(O.unpack(o).values()), o, equal_nan=True)


# --------------------------------------------------------------------------
# the task
# --------------------------------------------------------------------------

def test_registration():
    e = gym.make("AirCombat/Circular-v0")
    assert e.action_space.n == 9                    # heading 3 x speed 3
    import aircombat_gym.wvr.envs as t
    t._register()                                   # importing twice is harmless


def test_the_altitude_channel_is_closed():
    """Assignment 01 removes the vertical.  If it leaks back in, the task gets
    its difficulty back (ace 0.97 co-altitude against 0.30 at +-5,000 ft) and
    the measured baselines stop applying."""
    e = _env()
    assert ALTITUDE not in e.channels
    for a in range(e.action_space.n):
        assert e.decode(a)[2] == 0.0, f"action {a} moves the altitude target"


def test_both_aircraft_start_co_altitude():
    for seed in range(20):
        d = O.unpack(_env(**{})._obs(_env()._combat.observe()))
        e = CircularTargetEnv()
        o, _ = e.reset(seed=seed)
        d = O.unpack(o)
        assert abs(d["own_h"] - d["opp_h"]) < 1.0, f"seed {seed} is not level"


@pytest.mark.parametrize("mode", ["discrete", "continuous"])
def test_action_round_trip(mode):
    e = _env(action_mode=mode)
    from aircombat_gym.wvr.actions import DELTA_HEADING_DEG, DELTA_SPEED_KT
    for h in DELTA_HEADING_DEG:
        for v in DELTA_SPEED_KT:
            got = e.decode(e.encode_action((h, v, 0.0)))
            assert got == pytest.approx((h, v, 0.0)), f"{mode}: {(h, v)} -> {got}"


def test_the_two_action_modes_reach_the_same_corners():
    """`continuous` has to be the convex hull of `discrete`, or a comparison
    between them is measuring reach rather than resolution."""
    d, c = _env(action_mode="discrete"), _env(action_mode="continuous")
    grid = {d.decode(a) for a in range(d.action_space.n)}
    hull = {c.decode(np.array(s, dtype=np.float32))
            for s in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]}
    assert hull <= grid


# --------------------------------------------------------------------------
# no reward (D10)
# --------------------------------------------------------------------------

def test_the_task_returns_zero_reward():
    e = _env()
    for a in range(9):
        assert e.step(a)[1] == 0.0


def test_the_package_computes_no_reward():
    """Shipping the gym must not ship the answer key.  An AST walk rather than
    a grep, so a `reward` spelled any other way is still caught."""
    offenders = []
    for path in PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.FunctionDef):
                name = node.name
            elif isinstance(node, (ast.Name, ast.Attribute)):
                name = getattr(node, "id", None) or getattr(node, "attr", None)
            if name and "reward" in name.lower():
                offenders.append(f"{path.relative_to(PKG)}:{node.lineno} {name}")
    assert not offenders, "the package names a reward:\n  " + "\n  ".join(offenders)


# --------------------------------------------------------------------------
# the terminal is reachable
# --------------------------------------------------------------------------

def _fly(agent, env, seeds):
    kills, ts = 0, []
    for s in seeds:
        _, info = env.reset(seed=s)
        agent.reset()
        while True:
            _, _, term, trunc, info = env.step(
                env.encode_action(agent.act(info)))
            if term or trunc:
                kills += bool(info.get("won"))
                if info.get("won"):
                    ts.append(info["t"])
                break
    return kills, (sum(ts) / len(ts) if ts else float("nan"))


def test_ace_can_actually_win():
    """A terminal nobody can reach is not a terminal.

    This replaces an arithmetic check on `DAMAGE_RATE` that was written for a
    geometry which no longer exists.  Flying is slower and it is the only
    version that cannot go stale: it fails if the cone, the lock, the damage
    rate, the spawn ring or the opponent's turn rate drift out of agreement.

    `ace` scores 29/30 on the tuned task; 6/8 here leaves room for noise while
    still failing loudly if the task becomes unwinnable.
    """
    kills, t_kill = _fly(Ace(), CircularTargetEnv(), range(900_000, 900_008))
    assert kills >= 6, f"ace only killed {kills}/8 -- is the task winnable?"
    assert t_kill < 90.0, f"kills take {t_kill:.0f}s; the clock is 120s"


def test_the_observation_stays_finite_under_random_play():
    """Unbounded is not the same as allowed to blow up.  Raw metres are fine;
    a NaN out of the FDM is not, and it would poison a replay buffer silently."""
    e = CircularTargetEnv()
    o, _ = e.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        o, r, term, trunc, _ = e.step(int(rng.integers(e.action_space.n)))
        assert np.isfinite(o).all(), "non-finite observation"
        assert r == 0.0
        if term or trunc:
            o, _ = e.reset()


def test_a_winning_policy_is_expressible_from_the_raw_observation():
    """Rebuild `ace` reading nothing but the 37 raw channels.

    This is the assignment's own `ObsWrapper` exercise, run as a test.  If a
    hand-derived transform scores what the bot scores off the referee's dict,
    then the raw spec carries everything the task needs -- so a policy that
    trains badly is a training problem, not an observation problem.  Worth
    knowing before anyone spends a night on hyperparameters.

    It also pins the arithmetic the student README describes: difference the
    positions, rebuild the velocities, lead by the time of flight.
    """
    from aircombat_gym.wvr.baselines import DV, _snap
    from aircombat_gym.wvr.engagement import MUZZLE_MS

    class ObsAce:
        """Everything below comes out of `obs.unpack`; nothing from `info`."""

        def reset(self):
            pass

        def act_from_obs(self, o):
            d = O.unpack(o)
            dx = d["opp_x"] - d["own_x"]
            dy = d["opp_y"] - d["own_y"]
            dz = d["opp_h"] - d["own_h"]
            r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
            dvx = d["opp_vx"] - d["own_vx"]
            dvy = d["opp_vy"] - d["own_vy"]
            dvz = d["opp_vz"] - d["own_vz"]
            tof = r / MUZZLE_MS
            ax, ay = dx + dvx * tof, dy + dvy * tof
            lead = (math.atan2(ax, ay) - d["own_psi"] + math.pi) % (2 * math.pi)
            lead -= math.pi
            r_dot = (dx * dvx + dy * dvy + dz * dvz) / r
            turn = _snap(math.degrees(lead) * 4.0, 30.0)
            if r > 900.0:
                dv = DV
            elif r_dot > -20.0 and r > 400.0:
                dv = 0.0
            else:
                dv = -DV
            return turn, dv, 0.0

    seeds = range(900_000, 900_008)
    ref, _ = _fly(Ace(), CircularTargetEnv(), seeds)

    env, bot, from_obs = CircularTargetEnv(), ObsAce(), 0
    for s in seeds:
        o, _ = env.reset(seed=s)
        while True:
            o, _, term, trunc, info = env.step(
                env.encode_action(bot.act_from_obs(o)))
            if term or trunc:
                from_obs += bool(info.get("won"))
                break

    assert from_obs >= ref - 2, (
        f"the raw observation loses something the bot relies on: "
        f"{from_obs}/8 from obs against {ref}/8 from the referee's dict")


def test_the_task_still_separates_the_baselines():
    """If `pursuit` matched `ace` the assignment would have no top end.  This is
    a floor on the ceiling, not a tight comparison -- n=8 cannot rank."""
    seeds = range(900_000, 900_008)
    ace_k, _ = _fly(Ace(), CircularTargetEnv(), seeds)
    pur_k, _ = _fly(Pursuit(), CircularTargetEnv(), seeds)
    assert ace_k >= pur_k, f"ace {ace_k}/8 is not ahead of pursuit {pur_k}/8"
