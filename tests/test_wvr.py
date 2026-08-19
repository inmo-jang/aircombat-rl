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
import pathlib

from aircombat_gym.wvr import actions as act
from aircombat_gym.wvr.baselines import LADDER as ALL_BOTS
from aircombat_gym.wvr.envs.circular import CircularTargetEnv
from aircombat_gym.wvr.baselines import Ace
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


def test_the_bench_and_the_tasks_share_one_weapon():
    """The keyboard bench must not have its own opinion about the gun.

    It had four: a 0.6 s lock against the tasks' 1.0, a 15 deg cone against 30,
    the three-factor damage model against a flat 0.33, and only one of the two
    guns modelled at all.  Each half was internally consistent, so flying by
    hand felt fine and taught a game nobody was graded on -- the same shape of
    failure as the two renderers that disagreed about the cone.
    """
    from aircombat_gym.tools import manual_operation as M
    from aircombat_gym.wvr.envs.base import DuelEnv

    src = inspect.getsource(M)
    assert "_bot_view" not in src.replace("`_bot_view`", ""), \
        "the bench is building its own view for the bots again"
    assert "TRACK_LOCK" not in src, "the bench has its own lock again"
    # and it reads the referee's numbers rather than restating them
    for name in ("health", "track", "wez_time", "eng", "cone_deg", "lock_s"):
        assert isinstance(getattr(M._Foe, name), property), \
            f"_Foe.{name} is not a view onto Combat"
    assert DuelEnv.track_lock == 1.0 and DuelEnv.flat_damage == 0.33


def test_every_bot_flies_in_the_keyboard_tool():
    """Every scripted opponent survives being flown by the bench.

    `_Foe` never called `bot.reset()`, so anything with state (`Circler`'s rate
    carry, `Evader`'s break timer) died on its first act() -- including
    `--enemy circler`, the command the student README documents.  The bots now
    read `Combat.observe()` directly, which is what makes "the same objects fly
    here" true rather than aspirational, but they still have to actually fly.
    """
    import math
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from aircombat_gym.core.aircraft import Aircraft
    from aircombat_gym.core.envelope import V_MAX_KT
    from aircombat_gym.tools import manual_operation as M
    from aircombat_gym.wvr.envs.base import (ARENA_R, SIDES, Combat, DuelEnv,
                                             Initial)

    for name, cls in sorted(M.OPPONENTS.items()):
        combat = Combat(t_max=1e9, track_lock=DuelEnv.track_lock,
                        flat_damage=DuelEnv.flat_damage,
                        wez_cone_deg=DuelEnv.wez_cone_deg,
                        arena_m=ARENA_R, armed=SIDES)
        combat.reset(Initial("bench", ((0.0, -2000.0), (0.0, 0.0)),
                             (0.0, 90.0), (450.0, 0.5 * V_MAX_KT)))
        foe = M._Foe(cls(), combat)
        for _ in range(60):
            info = combat.observe()
            combat.step({"red": (0.0, 0.0, 0.0), "blue": foe.act(info["blue"])})
            foe.timeline()
        assert foe.state.flyable, f"{name} left the bench in a broken state"
        assert combat.ac["red"].state.flyable, f"{name}: our aircraft broke"

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


def test_the_designed_size_is_unchanged_by_the_resize_arithmetic():
    """`Layout.resize(1300, 860)` must reproduce the hand-measured rectangles.

    The proportions became fractions so the window could be dragged; if the
    arithmetic drifts even a pixel at the design size, every screenshot and
    every number in this file's sibling docs is describing a different panel.
    """
    from aircombat_gym.tools import render

    render.Layout.resize(render.W, render.H)
    assert render.Layout.topdown == (0, 0, 880, 566)
    assert render.Layout.profile == (12, 580, 856, 268)
    assert render.Layout.cockpit == (880, 0, 420, 300)
    assert render.Layout.readout == (880, 300, 420, 560)


def test_the_panel_fits_whatever_shape_the_window_is():
    """Nothing overflows, and the readout keeps the width its text needs.

    Rendering a tall window is what found both of these: a stacked arrangement
    gave the readout 360 px of height and it overprinted itself, and stretching
    the columns gave it 331 px of width and the key help lost its last word.
    """
    from aircombat_gym.tools import render

    for w, h in ((1300, 860), (1024, 1200), (2048, 1280), (960, 540)):
        render.Layout.resize(w, h)
        for name in ("topdown", "profile", "cockpit", "readout"):
            x, y, rw, rh = getattr(render.Layout, name)
            assert x >= 0 and y >= 0, f"{name} off the canvas at {w}x{h}"
            assert x + rw <= w and y + rh <= h, f"{name} overflows at {w}x{h}"
        assert render.Layout.readout[2] >= render.READOUT_MIN_W,             f"readout too narrow for its text at {w}x{h}"
    render.Layout.resize(render.W, render.H)


def test_the_window_scales_instead_of_cropping():
    """Drag the window anywhere and the whole panel is still on screen.

    The tools draw onto a fixed 1300x860 canvas and `present()` fits it into
    whatever the window has become, so the aspect must survive and the fit must
    never overflow -- a panel cropped at the edge loses the readout column,
    which is where every number lives.
    """
    from aircombat_gym.tools import render

    cw, ch = render.Layout.W, render.Layout.H
    for size in ((cw, ch), (960, 1080), (1920, 1080), (640, 480), (2560, 880)):
        r = render.blit_rect(size)
        assert r.w <= size[0] and r.h <= size[1], f"overflows at {size}"
        assert abs(r.w / r.h - cw / ch) < 0.01, f"aspect lost at {size}"
        assert r.x >= 0 and r.y >= 0, f"off the top-left at {size}"
        # centred: the two margins differ by at most the odd pixel
        assert abs((size[0] - r.w) - 2 * r.x) <= 1
        assert abs((size[1] - r.h) - 2 * r.y) <= 1


def test_every_overlay_branch_draws():
    """`p`, a handover, a low warning, a kill -- each has its own overlay.

    None of them was covered, and a refactor that hoisted a coordinate into the
    first branch went unnoticed until someone pressed `p` and the window closed:
    the other branches ran with the name unbound.  The overlays are the only
    part of the panel with real control flow, so they are the part worth
    driving.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from aircombat_gym.tools import render, manual_operation as M
    from aircombat_gym.core.aircraft import Aircraft

    screen, fonts, _ = render.open_window("overlay test")
    ac = Aircraft()
    ac.reset(x=0.0, y=0.0, psi=0.0, v_kt=400.0)
    s = ac.state
    ctl = dict(aileron=0.0, elevator=0.0, rudder=0.0, throttle=0.5)
    meter = M._Meter()

    cases = {"running": dict(paused=False, crash=None, handover=None),
             "paused": dict(paused=True, crash=None, handover=None),
             "handover": dict(paused=False, crash=None, handover=(1.0, "STICK")),
             "crashed": dict(paused=False, crash=(9.0, 420.0, -30.0, 500.0),
                             handover=None)}
    for name, kw in cases.items():
        try:
            M._panel(screen, fonts, ac, s, ctl, (0.0, 0.0, 0.0), 8.0,
                     meter=meter, live=None, rtf=1.0, layer="AUTOPILOT",
                     cmd_mode="CONTINUOUS", throttle_cap=1.0, ramp_s=0.6, **kw)
        except NameError as e:                     # the bug this test exists for
            raise AssertionError(f"{name} overlay: {e}") from None
    pygame.quit()


def test_importing_the_package_does_not_pull_in_a_viewer():
    """`wvr.play` reaches up into `tools/` -- importing `wvr` must not.

    The hand-play harness reuses the renderer rather than growing a second one,
    which points the wrong way through the layers: `tools` is built on `wvr`.
    That is affordable exactly because `wvr/__init__` does not import `play`,
    so the dependency only exists while the script runs.  Asserted rather than
    asserted-in-a-comment: a stray top-level import in any env module would
    make every downstream user of the gym depend on pygame.
    """
    import subprocess
    import sys
    probe = ("import sys; import aircombat_gym.wvr.envs; "
             "print(int('pygame' in sys.modules), int('torch' in sys.modules))")
    import aircombat_gym
    repo = pathlib.Path(aircombat_gym.__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, cwd=str(repo))
    assert out.returncode == 0, out.stderr[-400:]
    pg, torch = out.stdout.split()[-2:]
    assert pg == "0", "importing the gym pulled in pygame"
    assert torch == "0", "importing the gym pulled in torch"


def test_every_environment_is_reachable_from_the_hand_play_harness():
    """`--env` must list all of them.  The module asserts it at import time;
    this makes the failure a test rather than a crash on someone's first run."""
    from aircombat_gym.wvr import play
    from aircombat_gym.wvr.envs import ENVS
    assert set(play.ENV_BY_NAME.values()) == set(ENVS)


def test_swapping_the_action_space_keeps_the_episode():
    """`m` in the hand-play harness changes the space and nothing else.

    The point of the toggle is to feel the same engagement under both action
    spaces, which is worthless if switching respawns it -- and a respawn would
    be easy to ship unnoticed, because the picture would still look plausible.
    """
    from aircombat_gym.wvr.envs.advantaged import AdvantagedFightEnv

    env = AdvantagedFightEnv()
    env.reset(seed=5)
    for _ in range(20):
        env.step(4)
    before = (env._combat.t, env._combat.ac["red"].state.x,
              env._combat.ac["blue"].state.y, env._combat.health["blue"])

    env.set_action_mode("continuous")
    assert env.action_space.shape == (2,)
    after = (env._combat.t, env._combat.ac["red"].state.x,
             env._combat.ac["blue"].state.y, env._combat.health["blue"])
    assert before == after, "the toggle disturbed the engagement"

    # and the new space is actually usable
    env.step(env.action_space.sample())
    env.set_action_mode("discrete")
    assert env.action_space.n == 9
    env.step(4)
