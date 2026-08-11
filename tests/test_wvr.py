"""The weapon and the scripted opponents: `aircombat_gym/wvr/`.

Ported from the 2D suite on 2026-08-10 with the modules they cover.  Most are
here because of a real incident rather than because the code looked risky:

  * `test_bots_are_behaviourally_distinct` -- a degenerate bot was once a
    byte-for-byte copy of `pursuit`, so the two tied exactly and an entire gate
    table meant nothing
  * `test_every_bot_flies_in_the_keyboard_tool` -- the tool and the env fed a
    bot two different dicts, and every bot with state crashed on its first frame
  * `test_the_vertical_is_in_the_weapon_model` -- `_kin` had vz inverted, which
    the 2D suite could not have caught

Named for the package directory it covers.  Observation spec and the
assignments themselves are in `test_envs.py`.
"""
from __future__ import annotations

import inspect
import math
import re

import numpy as np
import pytest

from aircombat_gym.wvr import actions as act
from aircombat_gym.wvr.baselines import ALL as ALL_BOTS
from aircombat_gym.wvr.envs.circular import CircularTargetEnv
from aircombat_gym.wvr.baselines import Ace, Bot, Circler, Lead, Pursuit
from aircombat_gym.wvr.engagement import WEZ_ATA_DEG, WEZ_R_MAX, WEZ_R_MIN, look


class _K:
    def __init__(self, x, y, h, psi):
        self.x, self.y, self.h, self.psi, self.theta = x, y, h, psi, 0.0
        self.vx, self.vy, self.vz = math.sin(psi) * 250, math.cos(psi) * 250, 0.0


# --------------------------------------------------------------------------
# the weapon
# --------------------------------------------------------------------------

def test_wez_needs_pointing_and_range_and_aspect():
    me = _K(0, 0, 6096, 0.0)
    assert look(me, _K(0, 600, 6096, 0.0)).in_wez              # dead six
    six = look(me, _K(0, 600, 6096, 0.0)).damage_rate
    nose = look(me, _K(0, 600, 6096, math.pi)).damage_rate
    assert nose < 0.5 * six, "aspect angle is not in the weapon model"
    assert not look(me, _K(0, WEZ_R_MAX + 200, 6096, 0.0)).in_wez
    assert not look(me, _K(0, WEZ_R_MIN - 50, 6096, 0.0)).in_wez
    assert not look(me, _K(600, 0, 6096, 0.0)).in_wez          # off the nose


def test_the_vertical_is_in_the_weapon_model():
    """The 2D version could not have caught this: `_kin` had vz inverted, so a
    target 1,000 ft above and one 1,000 ft below looked identical."""
    me = _K(0, 0, 6096, 0.0)
    level = look(me, _K(0, 600, 6096, 0.0))
    high = look(me, _K(0, 600, 6096 + 305, 0.0))               # +1,000 ft
    assert high.ata > level.ata, "altitude split does not move the gun line"
    assert not look(me, _K(0, 600, 6096 + 610, 0.0)).in_wez    # +2,000 ft


def test_wez_range_matches_the_turn_geometry():
    """R_MAX was 1,000 m against a 2.9 km turn circle: reachable 2.3 % of the
    time, and 168 matches produced zero kills.  Keep it tied to the geometry."""
    assert WEZ_R_MAX >= 1200.0
    assert WEZ_ATA_DEG <= 45.0


# A reachability test lived here and was removed on 2026-08-11.  It asserted
# `DAMAGE_RATE * dwell * factor >= 1.0` from the module constant, and the
# numbers came from a geometry that no longer exists (15 deg cone, +-5,000 ft
# altitude spread).  The rule it protected is still the right rule -- *a
# terminal nobody can reach is not a terminal* -- but arithmetic on a module
# constant is the wrong way to check it, and the constant is not even what the
# assignment uses (`Combat(flat_damage=...)` replaces the whole damage model).
# It comes back in `test_envs.py` as an empirical check: fly `ace` against the
# configured task and assert a kill actually happens.


# --------------------------------------------------------------------------
# bots
# --------------------------------------------------------------------------

def test_bots_are_behaviourally_distinct():
    """A degenerate bot was once a byte-for-byte copy of `pursuit`, which made
    the whole gate table meaningless -- the two tied exactly and nobody noticed."""
    bodies = {}
    for cls in ALL_BOTS:
        src = inspect.getsource(cls.act)
        bodies.setdefault(src.split("\n", 1)[1], []).append(cls.name)
    dupes = {k: v for k, v in bodies.items() if len(v) > 1}
    assert not dupes, f"identical bot behaviour: {list(dupes.values())}"


def test_every_bot_flies_in_the_keyboard_tool():
    """The tool and the env must hand a bot the same thing.

    Two real breakages, both silent until someone actually tried a bot the tool
    had never run.  `_Foe` never called `bot.reset()`, so anything with state
    (`Circler`'s rate carry, `Evader`'s break timer) died on its first act()
    -- including `--enemy circler`, the command the student README documents.
    And `_bot_view` omitted `lead_signed`, which is what `ace` steers on.
    """
    import math
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from aircombat_gym.core.aircraft import Aircraft
    from aircombat_gym.core.envelope import H0_FT, V_MAX_KT
    from aircombat_gym.tools import manual_operation as M
    
    for name, cls in sorted(M.OPPONENTS.items()):
        me = Aircraft(h0_ft=H0_FT)
        me.reset(x=0.0, y=-2000.0, psi=0.0, v_kt=450.0)
        foe = M._Foe(cls(), h0_ft=H0_FT, v_kt=0.5 * V_MAX_KT,
                     x=0.0, y=0.0, psi=math.pi / 2)
        for _ in range(60):
            foe.observe(me.step(0.0, 0.0, 0.0))
            foe.step(1.0 / 20.0)
        assert foe.state.flyable, f"{name} left the tool in a broken state"

    # The two dicts have to agree, and the direction matters.  Asserting only
    # `tool_keys <= duel_keys` -- "the tool invents nothing" -- is the wrong way
    # round: the breakage that actually happened was the tool *omitting*
    # `lead_signed`, which a subset check cannot see.  Check both.
    env = CircularTargetEnv()
    env.reset(seed=0)
    duel_keys = set(env._combat.observe()["blue"])
    tool_keys = set(foe._bot_view())
    assert tool_keys <= duel_keys, (
        f"the tool invents keys the env does not have: {tool_keys - duel_keys}")

    read = set()
    for cls in M.OPPONENTS.values():
        for src in (inspect.getsource(cls.act),
                    inspect.getsource(cls.reset)):
            read |= set(re.findall(r'info\["(\w+)"\]', src))
    for base in (Bot,):
        for m in ("_turn", "_match_alt"):
            read |= set(re.findall(r'info\["(\w+)"\]',
                                   inspect.getsource(getattr(base, m))))
    assert read <= tool_keys, (
        f"bots read keys the tool does not provide: {sorted(read - tool_keys)}")


def test_bots_emit_legal_three_channel_actions():
    """The heading grid went from five wide to three on 2026-08-11 and
    `DELTA_HEADING_DEG[-2]`, which two bots used for a "soft" turn, silently
    became 0.0.  Every bot has to land on the grid the policy shares."""
    env = CircularTargetEnv()
    env.reset(seed=0)
    info = env._combat.observe()
    for cls in ALL_BOTS:
        b = cls()
        b.reset()
        a = b.act(info["blue"])
        assert len(a) == 3, f"{cls.name} is not a 3-channel bot"
        assert a[0] in act.DELTA_HEADING_DEG, f"{cls.name} heading {a[0]}"
        assert a[1] in act.DELTA_SPEED_KT, f"{cls.name} speed {a[1]}"
        assert a[2] in act.DELTA_ALT_FT, f"{cls.name} altitude {a[2]}"


# --------------------------------------------------------------------------
# the game
# --------------------------------------------------------------------------

# Six tests were removed here on 2026-08-11 with `envs/gym_env.py`, the
# 18-channel `DogfightEnv` that `wvr/envs/base.py` replaced.  Four of them tested that
# wrapper and nothing else.  The two worth keeping moved to `test_envs.py`
# against the 37-channel spec: the observation stays finite, and a winning
# policy is expressible from the observation alone.
#
# One did *not* survive the move, and the reason is worth recording.
# `test_observation_is_symmetric_between_seats` asserted that a head-on start
# looks identical from both chairs -- true of the old egocentric encoding, and
# false of raw state, where the two aircraft have different absolute positions
# by construction.  That is a real consequence of the new spec rather than an
# oversight: seat symmetry now has to be produced by the student's transform,
# and self-play on a later rung will need it.


def test_the_same_seed_gives_the_same_fight():
    """Grading and every baseline in the write-up assume this."""
    def run():
        env = CircularTargetEnv()
        _, info = env.reset(seed=4242)
        bot = Ace()
        bot.reset()
        for _ in range(400):
            _, _, term, trunc, info = env.step(env.encode_action(bot.act(info)))
            if term or trunc:
                break
        return (round(env._combat.t, 6),
                round(env._combat.damage["red"], 9),
                info.get("outcome"))

    assert run() == run()


# --------------------------------------------------------------------------
# the learnability floor
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------

# Four tests lived here and were removed on 2026-08-11: they imported the
# reference reward totals from `training/lib/rewards.py`, and `training/` moved
# to `archived/` when the task spec changed (observation, action space and
# algorithm all replaced -- see project_01_circular.md 7).
#
# The rule they enforced is workplan 2.5 and is still load-bearing -- the first
# reference reward broke it by 12x, and the one before this rewrite by 5x.  It
# returns in `test_envs.py` against the new reward, where it belongs: the
# reward now lives in the answer code, not in the package.
