"""The flight-model bench.  Both control layers, one aircraft, `tab` to switch.

**This is not the tool for feeling the gym.**  `aircombat_gym.wvr.play` is: it
goes through `env.step()` with the task's own action space, so what you feel
there is what a policy is up against.  This bench is for the layer underneath --
the airframe, the FLCS, and the autopilot on top of them -- and most of what it
can do is something no policy can ask for.  Use it to answer "what is the
aircraft capable of, and what does the controller do with my command".

  STICK      you drive the control surfaces and the autopilot is bypassed.
             This is the f16 FLCS and nothing of ours.
  AUTOPILOT  you set (heading, speed, altitude) and the controller works out
             bank, g and throttle.  This is the interface a policy commands
             through, one level above the action space itself.

`tab` hands the aircraft over live, which is the test neither layer can run on
its own: put it inverted in a 60 deg dive with the stick, press tab, and see
whether the autopilot gets it back.  An untrained policy will reach attitudes
the autopilot's own test never leaves its envelope to find.

Three things about the stick surprise everyone, all measured:

  roll is a *rate* command.  `aileron-cmd-norm` feeds a roll-rate loop (full
      stick asks 180 deg/s, 196 measured), so holding the key does not settle at
      a bank angle -- it keeps rolling.  You set the bank and you take it off.
  it does not come back.  Centre the stick from 45 deg of bank and it rolls all
      the way to 175 -- inverted -- and stops there.  The bundled f16 holds
      6-7 deg of rudder at zero command, which makes 2 deg of sideslip, which
      rolls it; once inverted the sideslip washes out and it settles
      (JSBSim discussion #814).  Wings level is a job, not a default.
  pitch is a *g* command.  Elevator 0 is exactly 1 g in every flight condition
      (9 measured, trim elevator 0.0000), so centring holds the flight path
      angle you are on -- including straight down.

Rudder is nearly useless and feeling that is worth the keypress: five seconds of
full pedal moves the heading 1.6 deg, against 35-50 deg for the same time spent
banking.  The FLCS yaw damper cancels most of what the pilot asks for.

One key set for both layers.  The letters are the keypad's 3x3 block mirrored
onto the home row, so the aircraft flies one-handed from either -- laptops
without a keypad lose nothing.  Arrows and Home/PageUp/End/PageDown/Insert/
Delete work too: they are what the keypad sends with NumLock off.

      Q W E          7 8 9        rudder  /  vertical  /  rudder
      A S D    ==    4 5 6        left  /  neutral  /  right
      Z X C          1 2 3        throttle+  /  vertical  /  throttle-
                     0 .          (also throttle + / -)

  axis          STICK                    AUTOPILOT
  A / D  4 / 6  roll left / right        heading -/+
  W      8      pitch PUSH, nose DOWN    altitude UP, climb
  X      2      pitch pull, nose up      altitude down
  Z / C  1 / 3  throttle up / down       speed up / down
  Q / E  7 / 9  rudder left / right      (unused)
  S      5      centre the stick         freeze all three targets
  shift         skip the ramp            coarse steps (10 deg / 100 ft / 20 kt)
  m             --                       CONTINUOUS <-> DISCRETE

  tab             STICK <-> AUTOPILOT
  BACKSPACE       new random engagement (full reset)
  p / esc         pause / quit
  + - / g         zoom / trail length

The vertical axis is the one place the layers disagree, on purpose: W is
stick-forward in STICK (nose down) and the altitude target going up in AUTOPILOT.
Stick convention against autopilot convention -- they really are opposite, so
the panel spells out which one is live.

AUTOPILOT takes commands two ways and `m` toggles.  What differs is the grain,
and keeping both is most of the point: feeling how coarse the policy's grid is
*against* a control you can place exactly is what `wvr/play` cannot show you,
because it only has the grid.

  CONTINUOUS  keys move the setpoint itself, finely, and it stays where you left
              it -- turn strength is how far ahead you put it.  A tap nudges and
              holding sweeps: heading 2 deg / 40 deg/s, altitude 25 ft /
              500 ft/s, speed 5 kt / 100 kt/s.  A piloting aid; no policy has
              this interface.
  DISCRETE    keys emit the action deltas a policy emits, at the decision rate:
              heading +-30 deg, speed +-20 kt, altitude +-1000 ft.  Tapping
              lands the turn; holding re-derives the target from the current
              state every step, so the aircraft never catches up and the
              manoeuvre is sustained.  This is `wvr/actions.py` and nothing else.

`--tacview` serves TacView Advanced over TCP; it is Windows-only, so drop it on
macOS and Linux and the pygame panel is all of it.  `--acmi` records a file
anywhere and plays back in the free viewer.

`--enemy` puts a scripted opponent up, refereed by `envs.base.Combat` -- the
same weapon, the same lock and both barrels, so a shot that tells here tells
there.  The bench used to keep its own books and they had drifted apart.

  python -m aircombat_gym.tools.manual_operation --tacview --enemy circler
  python -m aircombat_gym.tools.manual_operation --mode stick --acmi flight.acmi
  python -m aircombat_gym.tools.manual_operation
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

import pygame

from ..core.aircraft import DECISION_HZ, SUBSTEPS, Aircraft
from ..core.control.autopilot import Autopilot, wrap_pi as _wrap
from ..wvr.engagement import MUZZLE_MS, WEZ_ATA_DEG, WEZ_R_MAX, WEZ_R_MIN
from ..wvr.baselines import BY_NAME as OPPONENTS
from ..wvr.envs.base import ARENA_R, SIDES, Combat, DuelEnv
from ..core.envelope import (ALT_MAX_FT, ALT_MIN_FT, G, H0_FT, N_STRUCT,
                        THROTTLE_CAP, V_MAX_KT, V_MIN_KT, in_measured_table,
                        level_turn_rate_deg_s, max_bank_deg, n_max)
from ..wvr.actions import DELTA_ALT_FT, DELTA_HEADING_DEG, DELTA_SPEED_KT
from ..core.tacview import AcmiFile, AcmiRealtime
from . import render
from .render import ACCENT, BAD, BG, DIM, FG, GOOD, WARN

# Geometry comes from `render.Layout` so this tool and `auto_operation` cannot
# drift apart.  The aliases keep the rest of the file readable.
PROFILE_WINDOW_S = render.PROFILE_WINDOW_S
# full scale for the energy bar: the ceiling plus the speed limit as height
EH_FULL_FT = 50000.0

HEIGHT_BLUE = (80, 120, 200)

TRAIL_LENGTHS = (600, 2400, 0)          # 0 = unlimited
LAYERS = ("STICK", "AUTOPILOT")
CMD_MODES = ("CONTINUOUS", "DISCRETE")

# --- stick: how fast a key press becomes deflection --------------------------
# A keyboard is on/off and a stick is not.  Ramping matters most on roll: it is
# a rate command, so without fine control you can only ask for 180 deg/s and
# will never stop on a bank angle.  Measured, holding the roll key then letting
# go at 450 kt (`--ramp 0.4`):
#
#     hold   stick reaches   settles at
#     0.05s      0.12          0.8 deg
#     0.10s      0.25          2.8
#     0.20s      0.50         11.2
#     0.30s      0.75         23.6
#     0.40s      1.00        keeps rolling
#
# So the whole useful range lives inside half a second of keypress, which is
# why it feels twitchy.  A slower ramp spreads the same range over more time; it
# does not make roll less sensitive, because the sensitivity is the rate command
# itself.  Nothing holds a bank angle here -- that is what AUTOPILOT is for.
# `--ramp` sets the seconds to full so the feel can be settled by flying it.
RAMP_TO_FULL_S = 0.40   # default; --ramp overrides
RAMP_DOWN = 5.0         # per second back to centre when released (spring)
THROTTLE_RATE = 0.40    # per second (shift: 3x) -- a lever, it stays put

ALPHA_SOFT_DEG = 13.0   # where the schedule has started to bite noticeably
ALPHA_HARD_DEG = 15.5   # measured plateau under full aft stick below 450 kt

# --- autopilot: CONTINUOUS-mode increments ------------------------------------
# Deliberately NOT the action table.  CONTINUOUS mode is a piloting aid and
# wants fine resolution per press with speed from key repeat; DISCRETE is the
# frozen action space and uses actions.DELTA_*.
#
# Sized against what the aircraft can do.  Autopilot asks for the full available
# turn rate at 30 deg of heading error and turns at at most ~13 deg/s, so at
# 20 Hz repeat 2 deg a press sweeps the target at 40 deg/s: under a second of
# held key outruns the aircraft and establishes a maximum-rate turn, while one
# tap is a 2 deg nudge, about 7 % of full rate, which fine tracking needs.
# Altitude the same way -- the climb-rate command saturates at 250 ft/s, reached
# at 500 ft of error, so a second of held key again asks for everything.
TGT_HEADING_DEG = 2.0
TGT_HEADING_BIG = 10.0
TGT_SPEED_KT = 5.0
TGT_SPEED_BIG = 20.0
TGT_ALT_FT = 25.0
TGT_ALT_BIG = 100.0
KEY_REPEAT_MS = (220, 50)           # delay, interval -> 20 Hz while held

# --- one key set, four axes, both layers -------------------------------------
# The two layers are the same three axes at different levels -- that is what the
# autopilot chain *is*: the heading loop drives bank, the altitude loop drives Nz,
# the speed loop drives throttle.  So a key moves the same axis whichever layer
# is flying, and only the level of abstraction changes.  Rudder is the one extra
# and it barely does anything (5 s of full pedal moves the heading 1.6 deg).
#
#   axis          STICK            AUTOPILOT       QWE/ASD/ZXC   keypad
#   lateral       roll             heading        A / D         4 / 6
#   vertical      pitch            altitude       W / X         8 / 2
#   longitudinal  throttle         speed          Z / C         1 / 3, 0 / .
#   rudder        rudder           --             Q / E         7 / 9
#   neutral       centre stick     freeze targets S             5
#
# The letters are the keypad's 3x3 block mirrored onto the home row, so the two
# hands learn one shape.  Arrows and Home/PageUp/End/PageDown/Insert/Delete come
# along free: they are what the keypad sends with NumLock off, and binding them
# to the same axes makes NumLock stop mattering.
#
# ONE ASYMMETRY, deliberate: the vertical axis is inverted in STICK only.
# W is stick-forward -- nose down, descend -- while in AUTOPILOT W raises the
# altitude nudge.  Stick convention against autopilot-nudge convention; they really
# are opposite, so the panel names the direction in both layers.
AXES = {
    "lateral": (
        (pygame.K_a, pygame.K_KP4, pygame.K_LEFT),                     # negative
        (pygame.K_d, pygame.K_KP6, pygame.K_RIGHT),                    # positive
    ),
    "vertical": (
        (pygame.K_x, pygame.K_KP2, pygame.K_DOWN),
        (pygame.K_w, pygame.K_KP8, pygame.K_UP),
    ),
    "longitudinal": (
        (pygame.K_c, pygame.K_KP3, pygame.K_KP_PERIOD,
         pygame.K_PAGEDOWN, pygame.K_DELETE),
        (pygame.K_z, pygame.K_KP1, pygame.K_KP0,
         pygame.K_END, pygame.K_INSERT),
    ),
    "rudder": (
        (pygame.K_q, pygame.K_KP7, pygame.K_HOME),
        (pygame.K_e, pygame.K_KP9, pygame.K_PAGEUP),
    ),
}
NEUTRAL_KEYS = (pygame.K_s, pygame.K_KP5, pygame.K_CLEAR, pygame.K_SPACE)


def _axis_held(keys, name: str) -> float:
    """-1 / 0 / +1 from whatever is held down on this axis."""
    neg, pos = AXES[name]
    return float(any(keys[k] for k in pos)) - float(any(keys[k] for k in neg))


def _axis_of(key) -> tuple[str | None, float]:
    """Which axis a single key press belongs to, and its sign."""
    for name, (neg, pos) in AXES.items():
        if key in pos:
            return name, 1.0
        if key in neg:
            return name, -1.0
    return None, 0.0


def _axis(cur: float, want: float, dt: float, snap: bool,
          rate: float = 1.0 / RAMP_TO_FULL_S) -> float:
    """One stick axis: ramp toward `want` in [-1, 0, 1], spring back to centre."""
    if want == 0.0:
        return cur - math.copysign(min(abs(cur), RAMP_DOWN * dt), cur or 1.0)
    if snap:
        return want
    return max(-1.0, min(1.0, cur + want * rate * dt))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=400.0, help="initial KTAS")
    ap.add_argument("--alt", type=float, default=H0_FT,
                    help="starting altitude, ft (the band is 5,000-30,000)")
    ap.add_argument("--mode", choices=[m.lower() for m in LAYERS],
                    default="autopilot", help="which layer to start in")
    ap.add_argument("--throttle-cap", type=float, default=THROTTLE_CAP,
                    help="autopilot throttle ceiling (D23 game balance knob, not "
                         "physics; 1.0 is full afterburner).  STICK is never "
                         "capped -- that layer is about the airframe")
    ap.add_argument("--enemy", choices=["none", *OPPONENTS], default="none",
                    help="put a target up.  'circler' holds 20,000 ft and half "
                         "top speed and turns one way forever -- rung 1 of the "
                         "ladder.  It carries the same gun you do")
    ap.add_argument("--enemy-range", type=float, default=2500.0,
                    help="how far ahead the target starts, metres")
    ap.add_argument("--seed", type=int, default=None,
                    help="make the session reproducible: the same seed gives "
                         "the same sequence of engagements, BACKSPACE by "
                         "BACKSPACE.  Without it the first start is the fixed "
                         "one straight ahead and the rest are random")
    ap.add_argument("--cone", type=float, default=WEZ_ATA_DEG,
                    help=f"firing cone half-angle, degrees (default "
                         f"{WEZ_ATA_DEG:.0f}, the weapon's own).  An assignment "
                         f"may widen it -- project 01 runs at 30, so "
                         f"`--cone 30` is how you hand-fly that geometry")
    ap.add_argument("--ramp", type=float, default=RAMP_TO_FULL_S,
                    help="STICK: seconds of held key to reach full deflection. "
                         "Raise it if roll feels twitchy, lower it if reversals "
                         "feel sluggish.  shift always skips the ramp")
    ap.add_argument("--scale", type=float, default=6.0, help="metres per pixel")
    ap.add_argument("--acmi", metavar="PATH", help="also write a TacView .acmi file")
    ap.add_argument("--tacview", action="store_true",
                    help="also serve real-time telemetry (TacView Advanced)")
    ap.add_argument("--tacview-port", type=int, default=42674)
    ap.add_argument("--tacview-wait", type=float, default=0.0,
                    help="block this long for TacView before starting; 0 just "
                         "flies, and TacView can attach whenever you open it")
    args = ap.parse_args(argv)

    # One referee for both aircraft, carrying the tasks' weapon rather than a
    # second opinion about it.  The aircraft are swapped in afterwards because
    # `Combat` builds plain ones and this bench wants its own altitude and its
    # own throttle cap -- the cap is what `--throttle-cap` is for, and STICK is
    # deliberately never capped.
    combat = Combat(t_max=1e9, track_lock=DuelEnv.track_lock,
                    flat_damage=DuelEnv.flat_damage,
                    wez_cone_deg=args.cone, arena_m=ARENA_R,
                    armed=SIDES if args.enemy != "none" else ("red",))
    combat.ac["red"] = Aircraft(
        h0_ft=args.alt,
        autopilot=Autopilot(h0_ft=args.alt, throttle_cap=args.throttle_cap))
    combat.ac["blue"] = Aircraft(
        h0_ft=H0_FT, autopilot=Autopilot(h0_ft=H0_FT, throttle_cap=1.0))
    ac = combat.ac["red"]
    ac.reset(v_kt=args.speed)
    if not ac.backend.trim_ok:
        print(f"warning: level trim did not converge at {args.speed} kt "
              f"/ {args.alt:.0f} ft", file=sys.stderr)

    rng = random.Random(args.seed)

    def _spawn_foe(me=None, randomise=False):
        """Put a target up.  `randomise` draws a fresh geometry, as the task does.

        Straight ahead every time is right for a first look and wrong for
        practice: you learn the one approach and nothing about converting an
        arbitrary start, which is the whole of `Circular`.  BACKSPACE draws
        a new one.
        """
        if args.enemy == "none":
            return None
        # With a seed the whole session is reproducible, first engagement
        # included -- otherwise run N and run N+1 would differ by one draw and
        # "same seed, same fight" would quietly not be true.
        if args.seed is not None:
            randomise = True
        bot = OPPONENTS[args.enemy]()
        # Half of V_MAX, which is 325 kt -- slow enough to be caught by anything
        # that can point, fast enough that the chase is not a formality.
        if not randomise or me is None:
            x, y, psi = 0.0, args.enemy_range, math.pi / 2
        else:
            brg = rng.uniform(-math.pi, math.pi)
            r = rng.uniform(1500.0, 4000.0)
            x, y = me.x + r * math.sin(brg), me.y + r * math.cos(brg)
            psi = rng.uniform(-math.pi, math.pi)
            if hasattr(bot, "rate"):        # Circler: a fresh turn rate too
                bot.rate = rng.uniform(1.0, 5.0) * rng.choice((-1.0, 1.0))
        combat.ac["blue"].reset(x=x, y=y, psi=psi, v_kt=0.5 * V_MAX_KT)
        combat._zero()
        return _Foe(bot, combat)

    foe = _spawn_foe()
    foe_trail: list[tuple[float, float]] = []

    sinks = []
    if args.acmi:
        sinks.append(AcmiFile(args.acmi))
        print(f"writing {args.acmi}")
    live = None
    if args.tacview:
        live = AcmiRealtime(port=args.tacview_port)
        print(f"TacView real-time telemetry: {live.address}")
        print("  in TacView: Record -> Real-time Telemetry -> that address")
        print("  attach whenever -- late viewers get the header and a re-introduction")
        if args.tacview_wait > 0:
            print(f"  waiting up to {args.tacview_wait:.0f}s ...", flush=True)
            print("  connected" if live.wait(args.tacview_wait)
                  else "  no connection yet; flying anyway")
        sinks.append(live)

    pygame.init()
    screen, _fonts, clock = render.open_window("flight test")
    pygame.key.set_repeat(*KEY_REPEAT_MS)       # drives the nudge sweep
    f_big, f, f_small = _fonts

    dt_decision = 1.0 / DECISION_HZ
    ramp_rate = 1.0 / max(args.ramp, 1e-3)
    accum = 0.0
    last_wall = time.perf_counter()
    rtf = 1.0

    layer = LAYERS.index(args.mode.upper())
    cmd_mode = 0
    # trim leaves the throttle where it needed it; start the lever there so a
    # handover to STICK does not lurch
    ctl = dict(aileron=0.0, elevator=0.0, rudder=0.0,
               throttle=ac.backend.controls["throttle_cmd"])
    trail: list[tuple[float, float]] = []
    profile: list[tuple[float, float]] = []
    trail_i = 0
    scale = args.scale
    paused = False
    act = (0.0, 0.0, 0.0)
    meter = _Meter()
    crash: tuple[float, float, float, float] | None = None
    kill_freeze = False
    handover: tuple[float, str] | None = None
    s = ac.state
    running = True

    def to_stick():
        """Autopilot -> stick: pick the lever up where the controller left it."""
        out = ac.last
        if out is not None:
            ctl.update(aileron=out.aileron, elevator=out.elevator,
                       rudder=out.rudder, throttle=out.throttle)

    def to_autopilot():
        """Stick -> autopilot: aim at what we already have, then let it fly.

        Without this the controller inherits whatever targets were last set and
        hauls for them from an attitude that has nothing to do with them.  The
        integrators go too -- they have been winding up against an error nobody
        was closing.
        """
        ac.psi_cmd = s.psi
        ac.v_cmd_kt = _clamp(s.v_kt, V_MIN_KT, V_MAX_KT)
        ac.alt_cmd_ft = _clamp(s.h_ft, ALT_MIN_FT, ALT_MAX_FT)
        ac.autopilot.reset()

    while running:
        now = time.perf_counter()
        frame_dt = min(now - last_wall, 0.25)          # clamp after a stall
        last_wall = now
        accum += frame_dt
        # Presses collected here and scaled after the event loop, once the real
        # shift state is known.  It cannot be read from `e.mod`: with NumLock on,
        # Windows fakes a shift release around every numpad key so the numeric
        # value gets through, so shift+keypad arrived with no modifier set and
        # the coarse increments silently did nothing -- while shift+arrows,
        # which skip that path, worked.  `get_pressed()` sees the physical key.
        pressed: list[tuple[str, float]] = []

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_TAB:
                    layer = 1 - layer
                    (to_autopilot if LAYERS[layer] == "AUTOPILOT" else to_stick)()
                    handover = (s.t, LAYERS[layer])
                elif e.key == pygame.K_p:
                    paused = not paused
                elif e.key == pygame.K_BACKSPACE:
                    # Full reset *and* a fresh random engagement, on one key.
                    # `r` used to do the reset and BACKSPACE the respawn; two
                    # keys for one idea, and `r` sat beside the rudder keys
                    # where it read as a control rather than a command.
                    s = ac.reset(v_kt=args.speed)
                    ctl.update(aileron=0.0, elevator=0.0, rudder=0.0,
                               throttle=ac.backend.controls["throttle_cmd"])
                    trail.clear()
                    profile.clear()
                    meter = _Meter()
                    crash = handover = None
                    foe = _spawn_foe(s, randomise=True)
                    foe_trail.clear()
                    kill_freeze = False
                elif e.key == pygame.K_g:
                    trail_i = (trail_i + 1) % len(TRAIL_LENGTHS)
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    scale = max(1.0, scale / 1.3)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    scale = min(60.0, scale * 1.3)
                elif e.key in NEUTRAL_KEYS:
                    if LAYERS[layer] == "STICK":
                        ctl.update(aileron=0.0, elevator=0.0, rudder=0.0)
                    else:
                        ac.psi_cmd, ac.v_cmd_kt = s.psi, s.v_kt
                        ac.alt_cmd_ft = _clamp(s.h_ft, ALT_MIN_FT, ALT_MAX_FT)
                elif LAYERS[layer] == "STICK":
                    pass                       # stick axes are read as held keys
                elif e.key == pygame.K_m:
                    cmd_mode = 1 - cmd_mode
                elif CMD_MODES[cmd_mode] == "CONTINUOUS":
                    name, sgn = _axis_of(e.key)
                    if name is not None:
                        pressed.append((name, sgn))

        keys = pygame.key.get_pressed()
        snap = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        nudge = (0.0, 0.0, 0.0)
        for name, sgn in pressed:
            if name == "lateral":
                nudge = (nudge[0] + sgn * (TGT_HEADING_BIG if snap
                                           else TGT_HEADING_DEG),
                         nudge[1], nudge[2])
            elif name == "longitudinal":
                nudge = (nudge[0],
                         nudge[1] + sgn * (TGT_SPEED_BIG if snap
                                           else TGT_SPEED_KT),
                         nudge[2])
            elif name == "vertical":
                nudge = (nudge[0], nudge[1],
                         nudge[2] + sgn * (TGT_ALT_BIG if snap else TGT_ALT_FT))
        if LAYERS[layer] == "STICK":
            # pitch is inverted here: the up key is stick-forward, nose down,
            # while our sign convention above the backend is positive = pull
            want = (_axis_held(keys, "lateral"),
                    -_axis_held(keys, "vertical"),
                    _axis_held(keys, "rudder"))
            d_thr = _axis_held(keys, "longitudinal")
        elif CMD_MODES[cmd_mode] == "DISCRETE":
            # every channel is three-valued now, so shift buys nothing here --
            # there is one magnitude per axis and the key either asks for it or
            # does not.  That is exactly what a policy chooses between.
            act = (DELTA_HEADING_DEG[-1] * _axis_held(keys, "lateral"),
                   DELTA_SPEED_KT[-1] * _axis_held(keys, "longitudinal"),
                   DELTA_ALT_FT[-1] * _axis_held(keys, "vertical"))
        else:
            act = nudge
            if nudge != (0.0, 0.0, 0.0) and not paused:
                ac.nudge_target(nudge[0], nudge[1], nudge[2])

        # Fixed-step sim on a wall-clock accumulator, so one sim second is one
        # real second however long a frame takes to draw.  Rendering used to set
        # the pace and could not hold 20 Hz, which made it run in slow motion.
        # Capped at 4 catch-up steps so a stall cannot spiral.
        if paused or crash is not None:
            accum = 0.0
        n_steps = 0
        # The fight is over when the target is down: stop the clock and let the
        # picture stand, the way the policy viewer does.  BACKSPACE draws the
        # next engagement, `r` resets everything.
        if foe is not None and foe.health <= 0.0:
            kill_freeze = True
            accum = 0.0
        while accum >= dt_decision and n_steps < 4 and not kill_freeze:
            if LAYERS[layer] == "STICK":
                # The stick ramps on sim time, not frame time, so the same key
                # held for the same seconds always gives the same deflection.
                ctl["aileron"] = _axis(ctl["aileron"], want[0], dt_decision,
                                       snap, ramp_rate)
                ctl["elevator"] = _axis(ctl["elevator"], want[1], dt_decision,
                                        snap, ramp_rate)
                ctl["rudder"] = _axis(ctl["rudder"], want[2], dt_decision,
                                      snap, ramp_rate)
                rate = THROTTLE_RATE * (3.0 if snap else 1.0)
                ctl["throttle"] = _clamp(ctl["throttle"] + d_thr * rate * dt_decision,
                                         0.0, 1.0)
                # 20 Hz zero-order hold: measured indistinguishable from 120 Hz
                # across seven stick inputs, so the decision rate is honest here
                for _ in range(SUBSTEPS):
                    ac.backend.set_controls(ctl["aileron"], ctl["elevator"],
                                            ctl["rudder"], ctl["throttle"])
                    ac.backend.run_one()
                nxt = ac.backend.state
            elif CMD_MODES[cmd_mode] == "DISCRETE":
                nxt = ac.step(act[0], act[1], act[2])
            else:
                nxt = ac.hold()

            if not nxt.flyable:
                # keep `s` -- the last state that was still an aircraft
                crash = (s.t, s.v_kt, math.degrees(s.gamma), s.h_ft)
                accum = 0.0
                break
            s = nxt
            accum -= dt_decision
            n_steps += 1
            trail.append((s.x, s.y))
            profile.append((s.t, s.h_ft))
            meter.update(s, ac.backend.controls)
            # `None` for our side: the three control modes above already flew it,
            # and two of them cannot be written as an action delta.  The referee
            # keeps score for both and flies only the target.
            combat.step({"red": None,
                         "blue": (foe.act(combat.observe()["blue"])
                                  if foe is not None else None)})
            if foe is not None:
                foe.timeline()
                foe_trail.append((foe.state.x, foe.state.y))
                if len(foe_trail) > TRAIL_LENGTHS[0]:
                    foe_trail.pop(0)
                for note in foe.notes:
                    render.emit_event(sinks, note)
                foe.notes.clear()
            render.emit_frame(sinks, s.t, s,
                              foe.state if foe is not None else None,
                              locked=bool(foe is not None and foe.eng
                                          and foe.eng.in_wez))
        if n_steps:
            lim = TRAIL_LENGTHS[trail_i]
            if lim and len(trail) > lim:
                del trail[:len(trail) - lim]
            cut = s.t - PROFILE_WINDOW_S
            while len(profile) > 2 and profile[0][0] < cut:
                profile.pop(0)
        rtf += 0.1 * (n_steps * dt_decision / max(frame_dt, 1e-6) - rtf)
        if paused:
            act = (0.0, 0.0, 0.0)
        if handover is not None and s.t - handover[0] > 4.0:
            handover = None

        # `frame()` only accepts while the physics is stepping; polling here
        # means a viewer can attach when paused or after a crash too
        render.poll_sinks(sinks)
        _draw(screen, (f_big, f, f_small), ac, s, ctl, act, trail, profile,
              scale, paused, meter, live, rtf, LAYERS[layer],
              CMD_MODES[cmd_mode], crash, handover, args.throttle_cap,
              args.ramp, foe, foe_trail)
        render.present(screen)
        clock.tick(60)

    for sink in sinks:
        sink.close()
    pygame.quit()
    print(f"altitude {meter.h_min:,.0f} - {meter.h_max:,.0f} ft   "
          f"peak |alpha| {meter.alpha:.1f} deg   peak |Nz| {meter.nz:.2f} g   "
          f"peak roll rate {meter.p:.0f} deg/s")
    if meter.limited_s > 0.5:
        print(f"alpha limiter was biting for {meter.limited_s:.1f} s "
              f"(pitch authority below 70 %)")
    if meter.overstress_s > 0.05:
        print(f"over {N_STRUCT:.0f} g for {meter.overstress_s:.1f} s "
              f"(peak {meter.nz:.2f}) -- the bundled f16 has no g limiter")
    if crash is not None:
        print(f"crashed at t = {crash[0]:.1f} s   {crash[1]:.0f} kt   "
              f"gamma {crash[2]:+.0f} deg   {crash[3]:.0f} ft")
    return 0


class _Foe:
    """The other aircraft's pilot, and a window onto the referee's books.

    It used to *be* the referee: its own health, its own track streak, its own
    WEZ bookkeeping, and its own hand-built dict for the bot to read.  All four
    were second copies of `envs.base.Combat`, and every one had drifted -- a
    0.6 s lock against the tasks' 1.0, a 15 deg cone against 30, the
    three-factor damage model against a flat 0.33, and only our gun modelled at
    all.  The bench was showing a different game from the one being graded,
    which is the failure this project keeps writing down.

    Now `Combat` runs both aircraft and keeps score, and this is a name for
    "the side I am not flying".  Note whose streak `track` reports: the referee
    books one per shooter, so ours is the one that damages the target ahead.
    """

    def __init__(self, bot, combat, side: str = "blue") -> None:
        # A bot's state lives in reset(), not __init__ -- `Circler` counts steps
        # for its turn period, `Evader` tracks a break timer.  Every env calls
        # this on every episode; this tool did not, so any bot with state
        # crashed on its first act().
        self.bot = bot
        self.bot.reset()
        self.combat = combat
        self.side = side
        self.mine = "red" if side == "blue" else "blue"
        self.notes = []                  # TacView timeline, drained by main
        self.dead_at = None
        self._was_in = False

    # --- the referee's numbers, not a second set ----------------------------
    @property
    def ac(self):
        return self.combat.ac[self.side]

    @property
    def state(self):
        return self.ac.state

    @property
    def health(self) -> float:
        return self.combat.health[self.side]

    @property
    def track(self) -> float:
        """*Our* unbroken seconds on him -- the streak that damages this side."""
        return self.combat.track[self.mine]

    @property
    def wez_time(self) -> float:
        return self.combat.wez_time[self.mine]

    @property
    def eng(self):
        """Our engagement onto him, as the referee computed it."""
        return getattr(self.combat, "_eng", {}).get(self.mine)

    @property
    def cone_deg(self) -> float:
        return self.combat.wez_cone_deg

    @property
    def lock_s(self) -> float:
        return self.combat.track_lock

    @property
    def acmi_colour(self) -> str:
        # TacView cannot draw a weapon cone, so the target says it instead
        if self.health <= 0.0:
            return "Grey"
        if self.track >= self.lock_s:
            return "Orange"         # rounds landing
        return "Red"

    def act(self, info):
        """One decision from the bot, off the referee's own dict.

        `info` is `Combat.observe()[self.side]`, so the bots that fight in the
        env are these same objects reading these same keys.  There is no second
        view to keep in step, which is what `_bot_view` used to be and what it
        used to get wrong.
        """
        if self.health <= 0.0:
            return None             # dead: `Combat` holds it, momentum and all
        return self.bot.act(info)

    def timeline(self) -> None:
        """Turn the referee's state into TacView notes.  Edges only."""
        if self.health <= 0.0:
            if self.dead_at is None:
                self.dead_at = self.state.t
                self.notes.append("SPLASH")
            return
        e = self.eng
        if e is None:
            return
        if e.in_wez and not self._was_in:
            self.notes.append("IN WEZ")
            self._was_in = True
        elif not e.in_wez and self._was_in:
            self.notes.append("track broken")
            self._was_in = False


class _Meter:
    """Energy, rates and worst-cases -- everything the panel reads back.

    Ps = d/dt(h + V^2/2g) is the rate total energy is changing.  What it
    differentiates -- energy height -- is a *length*: the altitude reachable by
    trading away all the speed.  With altitude open that reading means what it
    says, and the split matters as much as the total, because n_max falls by
    nearly half between 5,000 and 30,000 ft.  Energy kept as height buys a
    better wing when it is cashed in.
    """

    TAU = 0.6          # s, smoothing on the rates
    GHOST_LAG = 5.0    # s, how far back the ghost marker trails

    def __init__(self) -> None:
        self.ps_fps = 0.0
        self.turn_rate = 0.0
        self.e_kin_ft = 0.0
        self.e_pot_ft = 0.0
        self.e_tot_ft = 0.0
        self.e_ghost_ft = 0.0
        self.alpha = 0.0
        self.nz = 0.0
        self.p = 0.0
        self.gamma = 0.0
        self.h_min = 1e9
        self.h_max = 0.0
        self.limited_s = 0.0
        self.overstress_s = 0.0
        self._prev = None
        self._hist: list[tuple[float, float]] = []

    def update(self, s, c) -> None:
        if self._prev is not None:
            pt, ph, pv, ptrack = self._prev
            dt = s.t - pt
            if dt > 1e-6:
                ps = ((s.h - ph) + (s.v ** 2 - pv ** 2) / (2.0 * G)) / dt / 0.3048
                dpsi = (s.track - ptrack + math.pi) % (2 * math.pi) - math.pi
                a = min(1.0, dt / self.TAU)
                self.ps_fps += a * (ps - self.ps_fps)
                self.turn_rate += a * (math.degrees(dpsi / dt) - self.turn_rate)
        self._prev = (s.t, s.h, s.v, s.track)

        self.e_kin_ft = (s.v ** 2) / (2.0 * G) / 0.3048
        self.e_pot_ft = s.h_ft
        self.e_tot_ft = self.e_pot_ft + self.e_kin_ft
        self._hist.append((s.t, self.e_tot_ft))
        cut = s.t - self.GHOST_LAG
        while len(self._hist) > 1 and self._hist[0][0] < cut:
            self._hist.pop(0)
        self.e_ghost_ft = self._hist[0][1]

        # magnitudes: pushing puts alpha negative and the schedule cuts pitch
        # authority just the same, so a signed max would read 0 through a bunt
        self.alpha = max(self.alpha, abs(math.degrees(s.alpha)))
        self.nz = max(self.nz, abs(s.nz))
        self.p = max(self.p, abs(math.degrees(s.p)))
        self.gamma = max(self.gamma, abs(math.degrees(s.gamma)))
        self.h_min = min(self.h_min, s.h_ft)
        self.h_max = max(self.h_max, s.h_ft)
        if c["pitch_authority"] < 0.70 and abs(c["elevator_cmd"]) > 0.2:
            self.limited_s += 1.0 / DECISION_HZ
        if abs(s.nz) > N_STRUCT:
            self.overstress_s += 1.0 / DECISION_HZ

    def accel_kt_s(self, v_kt: float) -> float:
        """Ps as the speed change it makes if none of it goes to height."""
        v = max(v_kt, 1.0) * 0.514444
        return G * (self.ps_fps * 0.3048) / v / 0.514444


# --- drawing -----------------------------------------------------------------

def _draw(screen, fonts, ac, s, ctl, act, trail, profile, scale, paused,
          meter, live, rtf, layer, cmd_mode, crash, handover, throttle_cap,
          ramp_s, foe=None, foe_trail=()):
    f_big, f, f_small = fonts
    screen.fill(BG, render.Layout.world)

    # The world views come from `tools/render.py`, shared with the policy
    # viewer.  They used to be duplicated here, and the copy read the firing
    # cone from a module constant while the task it drew ran at a different
    # one -- the picture was wrong in the only number that decides the game.
    own = render.Track(s, list(trail), list(profile))
    other = None
    if foe is not None:
        other = render.Track(foe.state, list(foe_trail), [],
                             dead=foe.health <= 0.0)
    weapon = render.Weapon(cone_deg=(foe.cone_deg if foe is not None
                                     else WEZ_ATA_DEG), r_min=WEZ_R_MIN,
                           r_max=WEZ_R_MAX, muzzle_ms=MUZZLE_MS,
                           lock_s=(foe.lock_s if foe is not None
                                   else DuelEnv.track_lock))
    in_wez = bool(foe is not None and foe.eng is not None and foe.eng.in_wez)
    lock = (min(1.0, foe.track / foe.lock_s) if foe is not None else 0.0)

    render.topdown(screen, render.Layout.topdown, f_small, own, other, weapon,
                   scale=scale, in_wez=in_wez, lock_frac=lock,
                   autopilot=(dict(psi_cmd=ac.psi_cmd, v_cmd_kt=ac.v_cmd_kt)
                             if layer == "AUTOPILOT" else None))
    render.profile(screen, render.Layout.profile, f_small, own,
                   alt_cmd_ft=ac.alt_cmd_ft if layer == "AUTOPILOT" else None)
    render.cockpit(screen, render.Layout.cockpit, (f, f_small),
                   own, other, weapon, in_wez=in_wez, lock_frac=lock)
    _panel(screen, fonts, ac, s, ctl, act, scale, paused, meter, live, rtf,
           layer, cmd_mode, crash, handover, throttle_cap, ramp_s, foe)









def _panel(screen, fonts, ac, s, ctl, act, scale, paused, meter, live, rtf,
           layer, cmd_mode, crash, handover, throttle_cap, ramp_s, foe=None):
    """Build the readout's *content*.  `render.Readout` decides where it lands.

    This used to lay itself out, adding pixel heights by hand, and it broke the
    moment the cockpit view took 300 px off the panel -- the instruments carried
    on drawing straight over the key help.  Now the panel is a list and the
    renderer fits it; adding a row cannot push anything off the bottom.
    """
    f_big, f, f_small = fonts
    c = ac.backend.controls
    stick = layer == "STICK"
    alpha = math.degrees(s.alpha)
    nmax = n_max(s.v_kt, s.h_ft)
    measured = in_measured_table(s.v_kt, s.h_ft)
    T = render.text

    items = [T(f"FLIGHT TEST   {layer}", WARN if stick else ACCENT, big=True),
             T("tab switches layer -- autopilot is bypassed" if stick
               else f"tab switches layer -- input {cmd_mode} (m)",
               DIM, small=True)]

    if foe is not None:
        e, dead = foe.eng, foe.health <= 0.0
        hcol = GOOD if foe.health > 0.5 else (WARN if foe.health > 0.2 else BAD)
        items += [render.gap(6),
                  render.bar(f"TARGET {'DESTROYED' if dead else 'health'}",
                             foe.health, (90, 90, 90) if dead else hcol,
                             f"{foe.health:.2f}")]
        if e is not None:
            hot = e.in_wez and foe.track >= foe.lock_s
            items += [
                T(f"range {e.r:6,.0f} m   lead err "
                  f"{math.degrees(e.ata_lead):4.1f} deg",
                  GOOD if e.in_wez else FG, small=True),
                T(f"aspect {math.degrees(e.aa):4.0f} deg   closing "
                  f"{-e.r_dot:+5.0f} m/s", DIM, small=True),
                T(f"hold  {foe.track:4.2f} / {foe.lock_s:.2f} s"
                  + ("   HITTING" if hot else ("   IN ZONE" if e.in_wez else "")),
                  WARN if hot else (GOOD if e.in_wez else DIM), small=True)]

    items.append(render.gap(8))
    if stick:
        items.append(render.custom(
            lambda sc, x, y, w: render.stick_box(sc, f_small, x, y, 124, ctl),
            152))
    else:
        out = ac.last
        # how much of the available turn this target is asking for -- read off
        # the bank command each law produced, not re-derived from gains
        demand = min(1.0, (abs(math.tan(out.phi_cmd))
                           / max(math.tan(out.max_bank), 1e-6)
                           if out is not None and out.max_bank > 1e-6 else 0.0))
        psi_err = math.degrees(_wrap(ac.psi_cmd - s.psi))
        d_psi, d_v, d_h = act
        bits = [b for b in ((f"{d_psi:+.0f}d" if d_psi else ""),
                            (f"{d_v:+.0f}kt" if d_v else ""),
                            (f"{d_h:+.0f}ft" if d_h else "")) if b]
        items += [
            render.head("COMMAND -> ACTUAL"),
            T(f"hdg {math.degrees(ac.psi_cmd) % 360:5.1f} -> "
              f"{math.degrees(s.psi) % 360:5.1f} ({psi_err:+.0f})", FG, small=True),
            T(f"spd {ac.v_cmd_kt:5.0f} -> {s.v_kt:5.0f} "
              f"({s.v_kt - ac.v_cmd_kt:+.0f})", FG, small=True),
            T(f"alt {ac.alt_cmd_ft:6,.0f} -> {s.h_ft:6,.0f} "
              f"({s.h_ft - ac.alt_cmd_ft:+,.0f})", FG, small=True),
            T(f"asks {demand * 100:3.0f} % turn   action "
              f"{'  '.join(bits) or '--'}",
              WARN if demand > 0.95 else DIM, small=True)]

    items.append(render.gap(6))
    if stick:
        a_col = BAD if alpha >= ALPHA_HARD_DEG else (
            WARN if alpha >= ALPHA_SOFT_DEG else FG)
        auth = c["pitch_authority"]
        items += [
            render.head("PITCH -- where your command goes"),
            render.marked_bar(f"alpha {alpha:6.1f} deg (peak {meter.alpha:.1f})",
                              abs(alpha) / 20.0, a_col, ALPHA_HARD_DEG / 20.0),
            T(f"authority {auth * 100:4.0f} %  what the schedule leaves",
              GOOD if auth > 0.85 else (WARN if auth > 0.6 else BAD), small=True),
            T(f"limiter {c['pitch_limiter']:+5.2f}   you ask "
              f"{c['elevator_cmd']:+5.2f} -> {c['pitch_net']:+5.2f}",
              DIM if c["pitch_limiter"] < 0.2 else WARN, small=True)]
    else:
        # bank and pitch are in the cockpit view now; these are the numbers it
        # cannot show to a tenth of a degree
        items += [
            T(f"alpha {alpha:+5.1f}   bank {math.degrees(s.phi):+6.1f}"
              f"   gamma {math.degrees(s.gamma):+5.1f}", FG, small=True),
            T(f"climb {s.h_dot / 0.3048:+6.0f} ft/s  limit 250", DIM, small=True)]

    n_col = BAD if abs(s.nz) > N_STRUCT else (
        WARN if s.nz > nmax * 0.97 else FG)
    note = ("OVER LIMIT" if meter.overstress_s > 0.05 else
            ("? off the measured table" if not measured
             else f"peak {meter.nz:.2f}"))
    turn_cap = level_turn_rate_deg_s(s.v_kt, s.h_ft)
    turn_low = level_turn_rate_deg_s(s.v_kt, ALT_MIN_FT)
    items += [
        render.gap(6),
        render.marked_bar(f"Nz {s.nz:5.2f} g of {nmax:.2f}   {note}",
                          abs(s.nz) / N_STRUCT, n_col,
                          min(1.0, nmax / N_STRUCT)),
        render.marked_bar(f"turn cap {turn_cap:5.1f} deg/s   "
                          f"now {abs(meter.turn_rate):4.1f}",
                          turn_cap / 18.0, WARN if turn_cap < 9 else GOOD,
                          min(1.0, turn_low / 18.0), f"{turn_low:.1f} at 5k"),
        render.bar(f"speed {s.v_kt:5.0f} kt   max bank "
                   f"{max_bank_deg(s.v_kt, s.h_ft):4.1f}",
                   (s.v_kt - V_MIN_KT) / (V_MAX_KT - V_MIN_KT), ACCENT)]

    ps = meter.ps_fps
    p_col = GOOD if ps > 5 else (BAD if ps < -5 else FG)
    items += [
        render.gap(6),
        render.marked_bar("energy  Eh = h + V^2/2g",
                          _clamp(meter.e_tot_ft / EH_FULL_FT, 0.0, 1.0),
                          HEIGHT_BLUE,
                          _clamp(meter.e_ghost_ft / EH_FULL_FT, 0.0, 1.0),
                          f"{meter.e_tot_ft:,.0f} ft"),
        T(f"Ps {ps:+5.0f} ft/s "
          f"{'gaining' if ps > 5 else ('BLEEDING' if ps < -5 else 'neutral')}"
          f"  = {meter.accel_kt_s(s.v_kt):+.1f} kt/s level", p_col, small=True)]

    def _inputs(sc, x, y, w):
        yy = y
        for label, cmd, deg in (("aileron ", c["aileron_cmd"], c["aileron_deg"]),
                                ("elevator", c["elevator_cmd"], c["elevator_deg"]),
                                ("rudder  ", c["rudder_cmd"], c["rudder_deg"])):
            col = BAD if abs(cmd) > 0.98 else FG
            sc.blit(f_small.render(label, True, DIM), (x, yy + 2))
            _bar2(sc, x + 58, yy, 96, 12, cmd, col)
            sc.blit(f_small.render(f"{cmd:+5.2f}", True, col), (x + 160, yy + 1))
            sc.blit(f_small.render(f"{deg:+6.1f}d", True, DIM), (x + 208, yy + 1))
            yy += 15
        thr, pos = c["throttle_cmd"], c["throttle_pos"]
        capped = (not stick) and thr >= throttle_cap - 1e-6
        sc.blit(f_small.render("throttle", True, DIM), (x, yy + 2))
        _bar(sc, x + 58, yy, 96, 12, thr, WARN if capped or pos > 1.0 else GOOD)
        cx = x + 58 + int(96 * throttle_cap)
        pygame.draw.line(sc, DIM if stick else FG, (cx, yy - 2), (cx, yy + 14),
                         1 if stick else 2)
        sc.blit(f_small.render(f"{thr:5.2f}  pos {pos:.2f}"
                               + (" AB" if pos > 1.0 else ""), True,
                               WARN if pos > 1.0 else FG), (x + 160, yy + 1))

    items += [render.gap(6), render.head("JSBSIM INPUTS  cmd-norm -> surface"),
              render.custom(_inputs, 66)]

    rt_col = GOOD if rtf > 0.92 else (WARN if rtf > 0.7 else BAD)
    if stick:
        keys = [T("W/8 NOSE DOWN  X/2 nose up  A D roll", WARN, small=True),
                T(f"Q E rudder  Z C throttle  S centre  ramp {ramp_s:.2f}s",
                  DIM, small=True)]
    else:
        keys = [T("W/8 CLIMB  X/2 descend  A D heading  Z C speed",
                  GOOD, small=True),
                T("S freeze  m CONT/DISC  shift coarse", DIM, small=True)]
    pinned = keys + [
        T("BACKSPACE new random engagement   p pause", DIM, small=True),
        T(f"tab layer  esc quit  + - zoom ({scale:.0f} m/px)  g trail",
          DIM, small=True),
        T(f"t = {s.t:6.1f} s   real time x{rtf:.2f}"
          + (f"   tacview {'up' if live.connected else 'waiting'}" if live else ""),
          rt_col, small=True)]

    render.Readout(screen, fonts).draw(items, pinned)

    # --- overlays on the map, not the panel ---------------------------------
    # Centre of the map column, which moves with the window.  Hoisted out of the
    # branches below because every one of them needs it and only the first one
    # used to set it -- pressing `p` raised NameError and closed the tool.
    cx = render.Layout.world[2] // 2
    if crash is not None:
        box = pygame.Rect(cx - 210, 26, 420, 58)
        pygame.draw.rect(screen, (60, 20, 24), box)
        pygame.draw.rect(screen, BAD, box, 2)
        screen.blit(f_big.render("CRASHED   BACKSPACE to restart", True, BAD),
                    (box.x + 16, box.y + 8))
        screen.blit(f_small.render(
            f"t {crash[0]:.1f} s   {crash[1]:.0f} kt   gamma {crash[2]:+.0f} deg"
            f"   {crash[3]:.0f} ft", True, FG), (box.x + 16, box.y + 36))
    elif handover is not None:
        screen.blit(f_big.render(f"-> {handover[1]}", True, GOOD),
                    (cx - 60, 26))
    elif s.h_ft < 2000:
        screen.blit(f_big.render("PULL UP", True, BAD), (cx - 44, 30))
    if foe is not None and foe.health <= 0.0:
        screen.blit(f_big.render("TARGET DOWN   BACKSPACE for a new one", True,
                                 GOOD), (cx - 210, 62))
    elif paused:
        screen.blit(f_big.render("PAUSED", True, WARN), (cx - 44, 62))


def _bar(screen, x, y, w, h, frac, col):
    frac = max(0.0, min(1.0, frac))
    pygame.draw.rect(screen, (45, 50, 62), (x, y, w, h))
    pygame.draw.rect(screen, col, (x, y, int(w * frac), h))


def _bar2(screen, x, y, w, h, frac, col):
    """Bipolar bar for a [-1, 1] channel, growing out from the centre."""
    frac = max(-1.0, min(1.0, frac))
    pygame.draw.rect(screen, (45, 50, 62), (x, y, w, h))
    mid = x + w // 2
    span = int((w // 2) * abs(frac))
    pygame.draw.rect(screen, col, (mid if frac >= 0 else mid - span, y, span, h))
    pygame.draw.line(screen, (150, 158, 175), (mid, y), (mid, y + h))


if __name__ == "__main__":
    raise SystemExit(main())
