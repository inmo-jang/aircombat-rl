"""Keyboard flight test.  Both control layers, one aircraft, `tab` to switch.

  STICK      you drive the control surfaces and guidance is bypassed.  This is
             the f16 FLCS and nothing of ours.
  GUIDANCE   you set (heading, speed, altitude) and the controller works out
             bank, g and throttle.  This is the interface a student policy sees.

Switching hands the aircraft over live, which is the test neither layer can run
on its own: put it inverted in a 60 deg dive with the stick, press tab, and see
whether guidance gets it back.  Gate C4 shakes the guidance layer with random
commands but can never reach that attitude, because it never leaves the
guidance envelope in the first place.  An untrained policy will.

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

  axis          STICK                    GUIDANCE
  A / D  4 / 6  roll left / right        heading -/+
  W      8      pitch PUSH, nose DOWN    altitude UP, climb
  X      2      pitch pull, nose up      altitude down
  Z / C  1 / 3  throttle up / down       speed up / down
  Q / E  7 / 9  rudder left / right      (unused)
  S      5      centre the stick         freeze all three targets
  shift         skip the ramp            coarse steps (10 deg / 100 ft / 20 kt)
  m             --                       CONTINUOUS <-> DISCRETE

  tab             STICK <-> GUIDANCE
  p / r / esc     pause / reset / quit
  + - / g         zoom / trail length

The vertical axis is the one place the layers disagree, on purpose: W is
stick-forward in STICK (nose down) and the altitude target going up in GUIDANCE.
Stick convention against autopilot convention -- they really are opposite, so
the panel spells out which one is live.

GUIDANCE takes commands two ways, `m` toggles.  Both are the human acting on the
aircraft; what differs is the grain, and the pair exists so a student can feel
what a continuous and a discrete action space are like before choosing one.

  CONTINUOUS  keys move the setpoint itself, finely, and it stays where you left
              it -- turn strength is how far ahead you put it.  A tap nudges and
              holding sweeps: heading 2 deg a press and 40 deg/s held, altitude
              25 ft and 500 ft/s, speed 5 kt and 100 kt/s.  (Aviation calls these
              markers "bugs"; the students here are not pilots, so the panel says
              target.)  Nothing in the gym exposes this as a policy interface --
              it is a piloting aid and a demonstration.
  DISCRETE    keys emit the action deltas a policy emits, at the decision rate:
              heading +-30 deg, speed +-20 kt, altitude +-1000 ft.  Tapping lands
              the turn; holding re-derives the target from the current state
              every step, so the aircraft never catches up and the manoeuvre is
              sustained.  This is `wvr/spaces.py` and nothing else.

`--tacview` serves TacView Advanced over TCP; it is Windows-only, so drop it on
macOS and Linux and the pygame panel is all of it.  `--acmi` records a file
anywhere and plays back in the free viewer.

  python -m aircombat_gym.tools.manual_operation --tacview --enemy circler
  python -m aircombat_gym.tools.manual_operation --mode stick --acmi flight.acmi
  python -m aircombat_gym.tools.manual_operation
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import pygame

from ..core.aircraft import DECISION_HZ, PHYSICS_HZ, SUBSTEPS, Aircraft
from ..core.control.guidance import Guidance
from ..wvr.engagement import (WEZ_ATA_DEG, WEZ_R_MAX, WEZ_R_MIN, look)
from ..wvr.baselines import BY_NAME as OPPONENTS
from ..core.envelope import (ALT_MAX_FT, ALT_MIN_FT, G, H0_FT, N_STRUCT,
                        THROTTLE_CAP, V_MAX_KT, V_MIN_KT, in_measured_table,
                        level_turn_rate_deg_s, max_bank_deg, n_max)
from ..wvr.spaces import DELTA_ALT_FT, DELTA_HEADING_DEG, DELTA_SPEED_KT
from ..core.tacview import AcmiFile, AcmiRealtime, state_to_object

W, H = 1300, 860
MAP_W = 880
TOP_H = 566                 # top-down view; the altitude strip sits under it
PROF_TOP = 574
PROF_H = H - PROF_TOP
PROFILE_WINDOW_S = 120.0
# full scale for the energy bar: the ceiling plus the speed limit as height
EH_FULL_FT = 50000.0

BG = (16, 18, 24)
PANEL = (24, 27, 36)
GRID = (34, 39, 52)
FG = (225, 230, 240)
DIM = (130, 140, 160)
ACCENT = (90, 200, 255)
WARN = (255, 170, 60)
BAD = (255, 90, 90)
GOOD = (120, 230, 150)
TRAIL = (70, 110, 150)
SKY = (58, 104, 158)
GROUND = (122, 88, 52)
HEIGHT_BLUE = (80, 120, 200)

TRAIL_LENGTHS = (600, 2400, 0)          # 0 = unlimited
LAYERS = ("STICK", "GUIDANCE")
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
# itself.  Nothing holds a bank angle here -- that is what GUIDANCE is for.
# `--ramp` sets the seconds to full so the feel can be settled by flying it.
RAMP_TO_FULL_S = 0.40   # default; --ramp overrides
RAMP_DOWN = 5.0         # per second back to centre when released (spring)
THROTTLE_RATE = 0.40    # per second (shift: 3x) -- a lever, it stays put

ALPHA_SOFT_DEG = 13.0   # where the schedule has started to bite noticeably
ALPHA_HARD_DEG = 15.5   # measured plateau under full aft stick below 450 kt

# --- guidance: CONTINUOUS-mode increments ------------------------------------
# Deliberately NOT the action table.  CONTINUOUS mode is a piloting aid and
# wants fine resolution per press with speed from key repeat; DISCRETE is the
# frozen action space and uses spaces.DELTA_*.
#
# Sized against what the aircraft can do.  Guidance asks for the full available
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
# guidance chain *is*: the heading loop drives bank, the altitude loop drives Nz,
# the speed loop drives throttle.  So a key moves the same axis whichever layer
# is flying, and only the level of abstraction changes.  Rudder is the one extra
# and it barely does anything (5 s of full pedal moves the heading 1.6 deg).
#
#   axis          STICK            GUIDANCE       QWE/ASD/ZXC   keypad
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
# W is stick-forward -- nose down, descend -- while in GUIDANCE W raises the
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
                    default="guidance", help="which layer to start in")
    ap.add_argument("--throttle-cap", type=float, default=THROTTLE_CAP,
                    help="guidance throttle ceiling (D23 game balance knob, not "
                         "physics; 1.0 is full afterburner).  STICK is never "
                         "capped -- that layer is about the airframe")
    ap.add_argument("--enemy", choices=["none", *OPPONENTS], default="none",
                    help="put a target up.  'circler' holds 20,000 ft and half "
                         "top speed and turns one way forever -- rung 1 of the "
                         "ladder.  It does not shoot back yet")
    ap.add_argument("--enemy-range", type=float, default=2500.0,
                    help="how far ahead the target starts, metres")
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

    ac = Aircraft(h0_ft=args.alt,
                  guidance=Guidance(h0_ft=args.alt, throttle_cap=args.throttle_cap))
    ac.reset(v_kt=args.speed)
    if not ac.backend.trim_ok:
        print(f"warning: level trim did not converge at {args.speed} kt "
              f"/ {args.alt:.0f} ft", file=sys.stderr)

    def _spawn_foe():
        if args.enemy == "none":
            return None
        # Half of V_MAX, which is 325 kt -- slow enough to be caught by anything
        # that can point, fast enough that the chase is not a formality.
        return _Foe(OPPONENTS[args.enemy](), h0_ft=H0_FT, v_kt=0.5 * V_MAX_KT,
                    x=0.0, y=args.enemy_range, psi=math.pi / 2)

    foe = _spawn_foe()
    foe_trail: list[tuple[float, float]] = []

    sinks = []
    if args.acmi:
        sinks.append(AcmiFile(args.acmi))
        print(f"writing {args.acmi}")
    live = None
    if args.tacview:
        live = AcmiRealtime(port=args.tacview_port)
        print(f"TacView real-time telemetry on {live.address} "
              f"(or 127.0.0.1:{args.tacview_port})")
        print("  in TacView: Record -> Real-time Telemetry -> enter that address")
        print("  attach whenever -- late viewers get the header and a re-introduction")
        if args.tacview_wait > 0:
            print(f"  waiting up to {args.tacview_wait:.0f}s ...", flush=True)
            print("  connected" if live.wait(args.tacview_wait)
                  else "  no connection yet; flying anyway")
        sinks.append(live)

    pygame.init()
    pygame.display.set_caption("aircombat - flight test")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    pygame.key.set_repeat(*KEY_REPEAT_MS)       # drives the nudge sweep
    f_big = pygame.font.SysFont("consolas", 22)
    f = pygame.font.SysFont("consolas", 16)
    f_small = pygame.font.SysFont("consolas", 13)

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
    handover: tuple[float, str] | None = None
    s = ac.state
    running = True

    def to_stick():
        """Guidance -> stick: pick the lever up where the controller left it."""
        out = ac.last
        if out is not None:
            ctl.update(aileron=out.aileron, elevator=out.elevator,
                       rudder=out.rudder, throttle=out.throttle)

    def to_guidance():
        """Stick -> guidance: aim at what we already have, then let it fly.

        Without this the controller inherits whatever targets were last set and
        hauls for them from an attitude that has nothing to do with them.  The
        integrators go too -- they have been winding up against an error nobody
        was closing.
        """
        ac.psi_cmd = s.psi
        ac.v_cmd_kt = _clamp(s.v_kt, V_MIN_KT, V_MAX_KT)
        ac.alt_cmd_ft = _clamp(s.h_ft, ALT_MIN_FT, ALT_MAX_FT)
        ac.guidance.reset()

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
                    (to_guidance if LAYERS[layer] == "GUIDANCE" else to_stick)()
                    handover = (s.t, LAYERS[layer])
                elif e.key == pygame.K_p:
                    paused = not paused
                elif e.key == pygame.K_r:
                    s = ac.reset(v_kt=args.speed)
                    ctl.update(aileron=0.0, elevator=0.0, rudder=0.0,
                               throttle=ac.backend.controls["throttle_cmd"])
                    trail.clear()
                    profile.clear()
                    meter = _Meter()
                    crash = handover = None
                    foe = _spawn_foe()
                    foe_trail.clear()
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
        while accum >= dt_decision and n_steps < 4:
            if foe is not None:
                foe.observe(s)
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
            objs = [state_to_object(s)]
            if foe is not None:
                foe.step(dt_decision)
                foe_trail.append((foe.state.x, foe.state.y))
                if len(foe_trail) > TRAIL_LENGTHS[0]:
                    foe_trail.pop(0)
                for note in foe.notes:
                    for sink in sinks:
                        sink.event(note, "101", "102")
                foe.notes.clear()
                objs.append(state_to_object(foe.state, "102", "F-16C",
                                            color=foe.acmi_colour))
                objs[0]["locked"] = "102" if foe.eng and foe.eng.in_wez else None
            for sink in sinks:
                sink.frame(s.t, objs)
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

        _draw(screen, (f_big, f, f_small), ac, s, ctl, act, trail, profile,
              scale, paused, meter, live, rtf, LAYERS[layer],
              CMD_MODES[cmd_mode], crash, handover, args.throttle_cap,
              args.ramp, foe, foe_trail)
        pygame.display.flip()
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
    """The other aircraft, plus the referee for shots taken at it.

    One-way for now: our gun is modelled, its gun is not.  That is a deliberate
    step -- a circling target that cannot shoot back is rung 1 of the ladder,
    the case where anything able to point and close should score, and it is the
    right thing to fly against before anything defends itself.

    Damage needs a *held* solution.  `TRACK_LOCK` seconds inside the envelope
    before rounds start telling, and the clock resets the moment the track
    breaks -- so a snapshot as the nose sweeps past is free, and breaking the
    other aircraft's aim is a real defence rather than a cosmetic one.
    """

    TRACK_LOCK = 0.6

    def __init__(self, bot, h0_ft: float, v_kt: float,
                 x: float, y: float, psi: float) -> None:
        self.bot = bot
        self.ac = Aircraft(h0_ft=h0_ft,
                           guidance=Guidance(h0_ft=h0_ft, throttle_cap=1.0))
        self.ac.reset(x=x, y=y, psi=psi, v_kt=v_kt)
        self.health = 1.0
        self.track = 0.0            # seconds of unbroken solution
        self.wez_time = 0.0
        self.damage = 0.0
        self.eng = None             # last Engagement, ours onto them
        self.notes: list[str] = []  # TacView timeline messages, drained by main
        self.dead_at: float | None = None
        self._was_in = False
        self._me = None

    @property
    def state(self):
        return self.ac.state

    @property
    def acmi_colour(self) -> str:
        # TacView cannot draw a weapon cone, so the target says it instead
        if self.health <= 0.0:
            return "Grey"
        if self.track >= self.TRACK_LOCK:
            return "Orange"         # rounds landing
        return "Red"

    def observe(self, me) -> None:
        self._me = me
        self.eng = look(_kin(me), _kin(self.state))

    def step(self, dt: float) -> None:
        """Fly the bot one decision step, then adjudicate our shot."""
        if self.health > 0.0:
            self.ac.step(*self.bot.act(self.state))
        else:
            self.ac.hold()          # a dead aircraft still has momentum
        if self.eng is None or self.health <= 0.0:
            return
        if self.eng.in_wez:
            self.wez_time += dt
            self.track += dt
            if not self._was_in:
                self.notes.append("IN WEZ")
                self._was_in = True
            if self.track >= self.TRACK_LOCK:
                d = self.eng.damage_rate * dt
                self.damage += d
                self.health = max(0.0, self.health - d)
                if self.health <= 0.0:
                    self.dead_at = self.state.t
                    self.notes.append("SPLASH")
        else:
            if self._was_in:
                self.notes.append("track broken")
                self._was_in = False
            self.track = 0.0


class _Kin:
    """What `engagement.look` needs: position, gun line, velocity."""

    __slots__ = ("x", "y", "h", "psi", "theta", "vx", "vy", "vz")

    def __init__(self, x, y, h, psi, theta, vx, vy, vz):
        self.x, self.y, self.h = x, y, h
        self.psi, self.theta = psi, theta
        self.vx, self.vy, self.vz = vx, vy, vz


def _kin(s) -> _Kin:
    # Horizontal speed is V*cos(gamma), not V -- in a 20 deg climb the
    # difference is 6 %, and altitude is a live axis now.  And vz is +h_dot:
    # dz in `look` is (foe.h - me.h) with up positive, so a climb is positive
    # here too.  The archived 2D version had -h_dot, which was invisible only
    # because the plane was locked and h_dot was always about zero.
    vh = s.v * math.cos(s.gamma)
    return _Kin(s.x, s.y, s.h, s.psi, s.theta,
                math.sin(s.track) * vh, math.cos(s.track) * vh, s.h_dot)


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
    screen.fill(BG, (0, 0, MAP_W, H))
    cx, cy = MAP_W // 2, TOP_H // 2

    def to_px(x, y):
        return int(cx + (x - s.x) / scale), int(cy - (y - s.y) / scale)

    step_px = 1000.0 / scale                                    # 1 km grid
    if step_px >= 12:
        gx = (-s.x / scale) % step_px
        while gx < MAP_W:
            pygame.draw.line(screen, GRID, (int(gx), 0), (int(gx), TOP_H))
            gx += step_px
        gy = (s.y / scale) % step_px
        while gy < TOP_H:
            pygame.draw.line(screen, GRID, (0, int(gy)), (MAP_W, int(gy)))
            gy += step_px

    if len(trail) > 1:
        # decimate: a few hundred segments look identical and cost a tenth
        stride = max(1, len(trail) // 400)
        pts = [to_px(x, y) for x, y in trail[::stride]]
        if len(pts) > 1:
            pygame.draw.lines(screen, TRAIL, False, pts, 2)

    if layer == "GUIDANCE":
        _speed_vector(screen, f_small, cx, cy, s, ac.v_cmd_kt)
        _heading_target(screen, cx, cy, ac.psi_cmd)
    else:
        # nose (where the gun points) against track (where it is going).  They
        # separate by up to 27 deg in hard manoeuvring -- bank rotates the angle
        # of attack into the horizontal plane -- and watching the two split is
        # half the reason to hand-fly this.
        _ray(screen, cx, cy, s.psi, 150, (90, 130, 90), 1)
        _ray(screen, cx, cy, s.track, 110, ACCENT, 1)

    if foe is not None:
        _wez(screen, cx, cy, s, foe, scale)
        _foe_symbol(screen, f_small, to_px, foe, foe_trail, scale)
        _pipper(screen, f_small, cx, cy, s, foe, scale)

    pts = []
    for ang, rad in ((0, 20), (2.5, 11), (math.pi, 5), (-2.5, 11)):
        a = s.psi + ang
        pts.append((cx + math.sin(a) * rad, cy - math.cos(a) * rad))
    pygame.draw.polygon(screen, ACCENT, pts)

    _profile(screen, f_small, profile, s,
             ac.alt_cmd_ft if layer == "GUIDANCE" else None, foe)
    _panel(screen, fonts, ac, s, ctl, act, scale, paused, meter, live, rtf,
           layer, cmd_mode, crash, handover, throttle_cap, ramp_s, foe)


def _profile(screen, font, profile, s, alt_cmd_ft, foe=None):
    """Altitude against time -- the axis a top-down view cannot show.

    Time rather than ground distance: in a hard turn the ground track doubles
    back on itself and the trace becomes unreadable, while what you want to see
    is how fast the height is being spent and bought back.
    """
    x0, y0, w, h = 12, PROF_TOP + 6, MAP_W - 24, PROF_H - 18
    pygame.draw.rect(screen, (20, 23, 30), (x0, y0, w, h))
    # Auto-scale.  A fixed 35,000 ft top flattened the trace against the ceiling
    # and read as the aircraft running out of climb -- it is not: the bundled
    # f16 still makes Ps = +77 ft/s at 55,000 ft on full AB.
    peak = max((ft for _, ft in profile), default=0.0)
    hi = max(35000.0, math.ceil((peak + 2000.0) / 5000.0) * 5000.0)

    def ypx(ft):
        return y0 + h - max(0.0, min(hi, ft)) / hi * h

    for ft, col, lbl in ((ALT_MAX_FT, (60, 70, 90), "30k ceiling"),
                         (ALT_MIN_FT, (60, 70, 90), "5k floor"),
                         (0.0, (110, 50, 50), "ground")):
        yy = int(ypx(ft))
        pygame.draw.line(screen, col, (x0, yy), (x0 + w, yy), 1)
        screen.blit(font.render(lbl, True, col), (x0 + 4, yy - 14))
    screen.blit(font.render(f"{hi / 1000:.0f}k", True, DIM), (x0 + 4, y0 + 2))
    if alt_cmd_ft is not None:
        yc = int(ypx(alt_cmd_ft))
        pygame.draw.line(screen, WARN, (x0, yc), (x0 + w, yc), 1)
        screen.blit(font.render(f"target {alt_cmd_ft:,.0f}", True, WARN),
                    (x0 + 4, yc + 2))

    if len(profile) > 1:
        t1 = profile[-1][0]
        t0 = min(profile[0][0], t1 - PROFILE_WINDOW_S)
        span = max(t1 - t0, 1e-3)
        stride = max(1, len(profile) // 600)
        pts = [(x0 + (t - t0) / span * w, ypx(ft)) for t, ft in profile[::stride]]
        if len(pts) > 1:
            pygame.draw.lines(screen, ACCENT, False, pts, 2)
        pygame.draw.circle(screen, FG, (int(pts[-1][0]), int(pts[-1][1])), 3)
    screen.blit(font.render(f"altitude, last {PROFILE_WINDOW_S:.0f} s"
                            f"   now {s.h_ft:,.0f} ft", True, DIM),
                (x0 + w - 240, y0 + 4))


def _wez(screen, cx, cy, s, foe, scale):
    """The gun envelope, drawn as the wedge it is.

    Only the horizontal slice of it -- the real zone is a cone in 3D and the
    referee uses the 3D angle, so a target directly above at the right range
    looks outside this wedge and still counts.  The panel carries the numbers
    that settle it.
    """
    lit = foe.eng is not None and foe.eng.in_wez
    col = GOOD if (lit and foe.track >= foe.TRACK_LOCK) else (
        (150, 200, 150) if lit else (58, 78, 62))
    r_in = WEZ_R_MIN / scale
    r_out = WEZ_R_MAX / scale
    for side in (-1, 1):
        a = s.psi + side * math.radians(WEZ_ATA_DEG)
        pygame.draw.line(screen, col,
                         (cx + math.sin(a) * r_in, cy - math.cos(a) * r_in),
                         (cx + math.sin(a) * r_out, cy - math.cos(a) * r_out), 1)
    for rr in (r_in, r_out):
        if rr > 3:
            rect = pygame.Rect(cx - rr, cy - rr, 2 * rr, 2 * rr)
            a0 = math.pi / 2 - s.psi - math.radians(WEZ_ATA_DEG)
            a1 = math.pi / 2 - s.psi + math.radians(WEZ_ATA_DEG)
            pygame.draw.arc(screen, col, rect, a0, a1, 1)


def _pipper(screen, font, cx, cy, s, foe, scale):
    """Where to point.  The gun needs lead, so this is not the target.

    Put the nose ray through the pipper and the solution is made.  Real
    aircraft draw exactly this and it is why leading a target is a skill rather
    than guesswork -- the first version of this tool asked for lead without
    showing it, and three minutes of hand flying produced no shots at all.
    """
    e = foe.eng
    if e is None or foe.health <= 0.0:
        return
    px = cx + e.aim_dx / scale
    py = cy - e.aim_dy / scale
    hot = e.in_wez and foe.track >= foe.TRACK_LOCK
    col = WARN if hot else (GOOD if e.in_wez else (150, 170, 200))
    pygame.draw.circle(screen, col, (int(px), int(py)), 9, 2)
    pygame.draw.circle(screen, col, (int(px), int(py)), 2)
    # tick from the target to the pipper, so the amount of lead is visible
    tx, ty = cx + (foe.state.x - s.x) / scale, cy - (foe.state.y - s.y) / scale
    pygame.draw.line(screen, (90, 100, 120), (tx, ty), (px, py), 1)


def _foe_symbol(screen, font, to_px, foe, trail, scale):
    st = foe.state
    if len(trail) > 1:
        stride = max(1, len(trail) // 300)
        pts = [to_px(x, y) for x, y in trail[::stride]]
        if len(pts) > 1:
            pygame.draw.lines(screen, (110, 60, 60), False, pts, 2)
    px, py = to_px(st.x, st.y)
    col = (120, 120, 120) if foe.health <= 0 else (
        WARN if foe.track >= foe.TRACK_LOCK else BAD)
    pts = []
    for ang, rad in ((0, 16), (2.5, 9), (math.pi, 4), (-2.5, 9)):
        a = st.psi + ang
        pts.append((px + math.sin(a) * rad, py - math.cos(a) * rad))
    pygame.draw.polygon(screen, col, pts)
    # a ring while the solution is held, so a hit is visible without reading
    if foe.track >= foe.TRACK_LOCK and foe.health > 0:
        pygame.draw.circle(screen, WARN, (px, py), 22, 2)
    if foe.eng is not None:
        screen.blit(font.render(f"{foe.eng.r:,.0f} m", True, col), (px + 18, py - 6))


def _adi(screen, font, x, y, r, s):
    """Attitude indicator.  Bank and pitch, which is what you fly a stick by.

    The world rotates and the aircraft symbol is fixed: right bank tilts the
    horizon's right end up, nose up pushes the horizon down.
    """
    d = 2 * r
    surf = pygame.Surface((d, d), pygame.SRCALPHA)
    surf.fill(SKY)

    px_per_deg = r / 26.0
    phi, theta = s.phi, math.degrees(s.theta)
    ux, uy = math.cos(phi), -math.sin(phi)          # along the horizon
    vx, vy = math.sin(phi), math.cos(phi)           # perpendicular, toward ground
    hx, hy = r + vx * theta * px_per_deg, r + vy * theta * px_per_deg

    L = 3 * r
    pygame.draw.polygon(surf, GROUND, [
        (hx - ux * L, hy - uy * L), (hx + ux * L, hy + uy * L),
        (hx + ux * L + vx * L, hy + uy * L + vy * L),
        (hx - ux * L + vx * L, hy - uy * L + vy * L)])
    pygame.draw.line(surf, (235, 240, 250), (hx - ux * L, hy - uy * L),
                     (hx + ux * L, hy + uy * L), 2)

    for m in range(-60, 61, 10):
        if m == 0:
            continue
        off = (theta - m) * px_per_deg
        mx, my = r + vx * off, r + vy * off
        half = (r * 0.30) if m % 20 == 0 else (r * 0.17)
        pygame.draw.line(surf, (215, 222, 235), (mx - ux * half, my - uy * half),
                         (mx + ux * half, my + uy * half), 1)
        if m % 20 == 0:
            surf.blit(font.render(f"{abs(m)}", True, (215, 222, 235)),
                      (mx + ux * half + 3, my + uy * half - 7))

    mask = pygame.Surface((d, d), pygame.SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), (r, r), r)
    surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    screen.blit(surf, (x, y))

    cx, cy = x + r, y + r
    pygame.draw.circle(screen, (70, 78, 96), (cx, cy), r, 2)
    pygame.draw.line(screen, WARN, (cx - 28, cy), (cx - 9, cy), 3)
    pygame.draw.line(screen, WARN, (cx + 9, cy), (cx + 28, cy), 3)
    pygame.draw.circle(screen, WARN, (cx, cy), 3)
    for tick in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
        a = math.radians(tick)
        ln = 9 if tick % 30 == 0 else 5
        pygame.draw.line(screen, DIM,
                         (cx + math.sin(a) * r, cy - math.cos(a) * r),
                         (cx + math.sin(a) * (r - ln), cy - math.cos(a) * (r - ln)), 1)
    # pointer moves the way the aircraft banks
    a = phi
    pygame.draw.polygon(screen, GOOD, [
        (cx + math.sin(a) * (r - 11), cy - math.cos(a) * (r - 11)),
        (cx + math.sin(a - 0.07) * (r - 24), cy - math.cos(a - 0.07) * (r - 24)),
        (cx + math.sin(a + 0.07) * (r - 24), cy - math.cos(a + 0.07) * (r - 24))])


def _stick_box(screen, font, x, y, size, ctl):
    """Where the ramped stick actually is, since the keys only say 'more'."""
    pygame.draw.rect(screen, (20, 23, 30), (x, y, size, size))
    pygame.draw.rect(screen, (60, 68, 84), (x, y, size, size), 1)
    mid = size // 2
    pygame.draw.line(screen, (44, 50, 62), (x, y + mid), (x + size, y + mid))
    pygame.draw.line(screen, (44, 50, 62), (x + mid, y), (x + mid, y + size))
    px = x + mid + ctl["aileron"] * (mid - 6)
    py = y + mid - ctl["elevator"] * (mid - 6)      # pull is up in the box
    pygame.draw.line(screen, (70, 90, 120), (x + mid, y + mid), (px, py), 1)
    pygame.draw.circle(screen, WARN, (int(px), int(py)), 5)
    screen.blit(font.render("pull", True, DIM), (x + mid - 12, y + 2))
    screen.blit(font.render("R", True, DIM), (x + size - 11, y + mid - 7))


def _speed_vector(screen, font, cx, cy, s, v_cmd_kt):
    """Current and commanded speed as a bar along the direction of travel.

    Length is speed, drawn on the track (where the aircraft is going, not where
    the nose points).  A caret sits at the commanded speed, so the gap between
    bar tip and caret is what the throttle is still working on.
    """
    def px(v_kt):
        frac = (v_kt - V_MIN_KT) / (V_MAX_KT - V_MIN_KT)
        return 34 + max(0.0, min(1.0, frac)) * 210

    ux, uy = math.sin(s.track), -math.cos(s.track)      # screen up is north
    nx, ny = uy, -ux                                    # perpendicular
    l_now, l_cmd = px(s.v_kt), px(v_cmd_kt)
    pygame.draw.line(screen, (44, 50, 62), (cx + ux * 34, cy + uy * 34),
                     (cx + ux * 244, cy + uy * 244), 3)
    col = WARN if v_cmd_kt > s.v_kt + 3 else (ACCENT if v_cmd_kt < s.v_kt - 3 else GOOD)
    tip = (cx + ux * l_now, cy + uy * l_now)
    pygame.draw.line(screen, col, (cx + ux * 34, cy + uy * 34), tip, 7)
    cxx, cyy = cx + ux * l_cmd, cy + uy * l_cmd
    pygame.draw.line(screen, FG, (cxx + nx * 12, cyy + ny * 12),
                     (cxx - nx * 12, cyy - ny * 12), 3)
    screen.blit(font.render(f"{v_cmd_kt:.0f}", True, FG),
                (cxx + nx * 16 - 8, cyy + ny * 16 - 7))
    screen.blit(font.render(f"{s.v_kt:.0f} kt", True, col),
                (tip[0] - nx * 20 - 14, tip[1] - ny * 20 - 7))


def _heading_target(screen, cx, cy, psi_cmd, radius=150):
    """Where the aircraft is being told to point."""
    bx = cx + math.sin(psi_cmd) * radius
    by = cy - math.cos(psi_cmd) * radius
    pygame.draw.line(screen, (70, 60, 30), (cx, cy), (bx, by), 1)
    pygame.draw.polygon(screen, WARN, [
        (bx + math.sin(psi_cmd) * 11, by - math.cos(psi_cmd) * 11),
        (bx + math.sin(psi_cmd + 2.6) * 8, by - math.cos(psi_cmd + 2.6) * 8),
        (bx + math.sin(psi_cmd - 2.6) * 8, by - math.cos(psi_cmd - 2.6) * 8)], 2)


def _panel(screen, fonts, ac, s, ctl, act, scale, paused, meter, live, rtf,
           layer, cmd_mode, crash, handover, throttle_cap, ramp_s, foe=None):
    f_big, f, f_small = fonts
    pygame.draw.rect(screen, PANEL, (MAP_W, 0, W - MAP_W, H))
    x0, y = MAP_W + 18, 12
    c = ac.backend.controls
    stick = layer == "STICK"

    def line(txt, col=FG, font=None, dy=18):
        nonlocal y
        screen.blit((font or f).render(txt, True, col), (x0, y))
        y += dy

    line(f"FLIGHT TEST   {layer}", WARN if stick else ACCENT, f_big, 24)
    line("tab switches layer -- guidance is bypassed" if stick
         else f"tab switches layer -- input {cmd_mode} (m)", DIM, f_small, 16)

    if foe is not None:
        e = foe.eng
        dead = foe.health <= 0.0
        line(f"TARGET  {'DESTROYED' if dead else 'health'}",
             DIM if not dead else GOOD, f_small, 15)
        hcol = GOOD if foe.health > 0.5 else (WARN if foe.health > 0.2 else BAD)
        _bar(screen, x0, y, 200, 11, foe.health, (90, 90, 90) if dead else hcol)
        y += 16
        if e is not None:
            hot = e.in_wez and foe.track >= foe.TRACK_LOCK
            line(f"range {e.r:6,.0f} m    lead err {math.degrees(e.ata_lead):4.1f} deg",
                 GOOD if e.in_wez else FG, f_small, 15)
            line(f"aspect {math.degrees(e.aa):4.0f} deg   closing "
                 f"{-e.r_dot:+5.0f} m/s", DIM, f_small, 15)
            line(f"hold  {foe.track:4.2f} / {foe.TRACK_LOCK:.2f} s"
                 + ("   HITTING" if hot else ("   IN ZONE" if e.in_wez else "")),
                 WARN if hot else (GOOD if e.in_wez else DIM), f_small, 18)
        y += 4

    _adi(screen, f_small, x0, y, 92, s)
    if stick:
        _stick_box(screen, f_small, x0 + 200, y + 30, 124, ctl)
    else:
        psi_err = math.degrees((ac.psi_cmd - s.psi + math.pi) % (2 * math.pi) - math.pi)
        # "how much of the available turn is this target asking for" -- each law
        # gets there differently, so read it off the bank command it produced
        # rather than re-deriving it from gains only one of them has
        out = ac.last
        demand = (abs(math.tan(out.phi_cmd)) / max(math.tan(out.max_bank), 1e-6)
                  if out is not None and out.max_bank > 1e-6 else 0.0)
        demand = min(1.0, demand)
        ty = y + 26
        for txt, col in (
                (f"hdg {math.degrees(ac.psi_cmd) % 360:5.1f}", DIM),
                (f" -> {math.degrees(s.psi) % 360:5.1f} ({psi_err:+.0f})", FG),
                (f"spd {ac.v_cmd_kt:5.0f}", DIM),
                (f" -> {s.v_kt:5.0f} ({s.v_kt - ac.v_cmd_kt:+.0f})", FG),
                (f"alt {ac.alt_cmd_ft:6,.0f}", DIM),
                (f" -> {s.h_ft:6,.0f} ({s.h_ft - ac.alt_cmd_ft:+,.0f})", FG),
                ("", DIM),
                (f"target asks {demand * 100:3.0f} % turn",
                 WARN if demand > 0.95 else DIM)):
            screen.blit(f_small.render(txt, True, col), (x0 + 200, ty))
            ty += 15
        d_psi, d_v, d_h = act
        bits = [b for b in ((f"{d_psi:+.0f}d" if d_psi else ""),
                            (f"{d_v:+.0f}kt" if d_v else ""),
                            (f"{d_h:+.0f}ft" if d_h else "")) if b]
        screen.blit(f_small.render("  ".join(bits) or "--", True, WARN),
                    (x0 + 200, ty))
    y += 196

    alpha = math.degrees(s.alpha)
    nmax = n_max(s.v_kt, s.h_ft)
    measured = in_measured_table(s.v_kt, s.h_ft)

    if stick:
        # --- the alpha limiter, read out of the FLCS rather than guessed -----
        line("PITCH -- where your command goes", DIM, f_small, 17)
        a_col = BAD if alpha >= ALPHA_HARD_DEG else (
            WARN if alpha >= ALPHA_SOFT_DEG else FG)
        line(f"alpha    {alpha:6.1f} deg   (peak {meter.alpha:.1f})", a_col)
        _bar(screen, x0, y, 200, 10, abs(alpha) / 20.0, a_col)
        for mark, col in ((ALPHA_SOFT_DEG, WARN), (ALPHA_HARD_DEG, BAD)):
            mx = x0 + int(200 * mark / 20.0)
            pygame.draw.line(screen, col, (mx, y - 2), (mx, y + 12), 1)
        y += 16
        auth = c["pitch_authority"]
        line(f"authority {auth * 100:4.0f} %  what the alpha schedule leaves",
             GOOD if auth > 0.85 else (WARN if auth > 0.6 else BAD), f_small, 15)
        line(f"limiter  {c['pitch_limiter']:+5.2f}  pushed back against you",
             DIM if c["pitch_limiter"] < 0.2 else WARN, f_small, 15)
        line(f"you ask  {c['elevator_cmd']:+5.2f}   elevator gets "
             f"{c['pitch_net']:+5.2f}", FG, f_small, 19)
    else:
        line("ATTITUDE", DIM, f_small, 17)
        line(f"alpha   {alpha:+6.1f} deg   bank {math.degrees(s.phi):+6.1f}")
        line(f"gamma   {math.degrees(s.gamma):+6.1f} deg   (peak "
             f"{meter.gamma:.0f})")
        line(f"climb   {s.h_dot / 0.3048:+6.0f} ft/s  limit 250", DIM, f_small, 19)

    line("G", DIM, f_small, 17)
    n_col = BAD if abs(s.nz) > N_STRUCT else (
        WARN if s.nz > nmax * 0.97 else FG)
    line(f"Nz      {s.nz:6.2f} g   of {nmax:.2f} available"
         f"{'' if measured else ' ?'}", n_col)
    _bar(screen, x0, y, 200, 10, abs(s.nz) / N_STRUCT, n_col)
    nx = x0 + int(200 * min(1.0, nmax / N_STRUCT))
    pygame.draw.line(screen, WARN if measured else DIM, (nx, y - 2), (nx, y + 12), 2)
    y += 16
    if meter.overstress_s > 0.05:
        # the bundled f16 has no g limiter of its own -- guidance plans against
        # N_STRUCT (D25) but the stick can simply exceed it
        line(f"peak {meter.nz:.2f} g   OVER {N_STRUCT:.0f} g FOR "
             f"{meter.overstress_s:.1f} s", BAD, f_small, 19)
    elif not measured:
        # n_max clamps at the edge of the measured grid instead of
        # extrapolating, so up here the reference has stopped moving
        line(f"peak {meter.nz:.2f} g   | frozen: outside the measured table",
             WARN, f_small, 19)
    else:
        line(f"peak {meter.nz:.2f} g   | = what this speed and height allow",
             DIM, f_small, 19)

    # --- what speed AND height are worth -------------------------------------
    mb = max_bank_deg(s.v_kt, s.h_ft)
    turn_cap = level_turn_rate_deg_s(s.v_kt, s.h_ft)
    turn_low = level_turn_rate_deg_s(s.v_kt, ALT_MIN_FT)
    line("WHAT THIS SPEED AND HEIGHT ALLOW", DIM, f_small, 17)
    line(f"speed   {s.v_kt:6.0f} kt   max bank {mb:4.1f}", DIM, f_small, 15)
    _bar(screen, x0, y, 200, 9, (s.v_kt - V_MIN_KT) / (V_MAX_KT - V_MIN_KT), ACCENT)
    y += 14
    line(f"turn cap {turn_cap:5.1f} deg/s   now {abs(meter.turn_rate):4.1f}",
         WARN if turn_cap < 9 else FG)
    _bar(screen, x0, y, 200, 10, turn_cap / 18.0, WARN if turn_cap < 9 else GOOD)
    _bar(screen, x0, y + 11, 200, 5, abs(meter.turn_rate) / 18.0, ACCENT)
    # the marker is the same speed flown at the floor: the gap is what the
    # altitude being held costs in turn performance right now
    lx = x0 + int(200 * min(1.0, turn_low / 18.0))
    pygame.draw.line(screen, (200, 120, 200), (lx, y - 3), (lx, y + 12), 2)
    y += 19
    line(f"| {turn_low:.1f} at the 5k floor -- what diving buys",
         DIM, f_small, 19)

    # --- energy: how much, where it is kept, which way it is going -----------
    line("ENERGY  Eh = h + V^2/2g", DIM, f_small, 17)
    pot = _clamp(meter.e_pot_ft / EH_FULL_FT, 0.0, 1.0)
    kin = _clamp(meter.e_kin_ft / EH_FULL_FT, 0.0, 1.0 - pot)
    pygame.draw.rect(screen, (45, 50, 62), (x0, y, 200, 15))
    pygame.draw.rect(screen, HEIGHT_BLUE, (x0, y, int(200 * pot), 15))
    pygame.draw.rect(screen, WARN, (x0 + int(200 * pot), y, int(200 * kin), 15))
    gx = x0 + int(200 * _clamp(meter.e_ghost_ft / EH_FULL_FT, 0.0, 1.0))
    pygame.draw.line(screen, FG, (gx, y - 3), (gx, y + 17), 2)
    y += 20
    line(f"{meter.e_tot_ft:6,.0f} ft  (5 s ago {meter.e_ghost_ft:,.0f})",
         GOOD if meter.e_tot_ft > meter.e_ghost_ft - 30 else BAD, f_small, 15)
    line(f"blue {meter.e_pot_ft:,.0f} height + orange {meter.e_kin_ft:,.0f} speed",
         DIM, f_small, 17)
    ps = meter.ps_fps
    p_col = GOOD if ps > 5 else (BAD if ps < -5 else FG)
    line(f"Ps      {ps:+6.0f} ft/s  "
         f"{'gaining' if ps > 5 else ('BLEEDING' if ps < -5 else 'neutral')}",
         p_col)
    line(f"        {meter.accel_kt_s(s.v_kt):+6.1f} kt/s if none goes to height",
         p_col, f_small, 18)

    # --- what actually reached JSBSim ----------------------------------------
    line("JSBSIM INPUTS   fcs/*-cmd-norm  ->  surface", DIM, f_small, 16)
    for label, cmd, deg in (("aileron ", c["aileron_cmd"], c["aileron_deg"]),
                            ("elevator", c["elevator_cmd"], c["elevator_deg"]),
                            ("rudder  ", c["rudder_cmd"], c["rudder_deg"])):
        col = BAD if abs(cmd) > 0.98 else FG
        screen.blit(f_small.render(label, True, DIM), (x0, y + 2))
        _bar2(screen, x0 + 58, y, 96, 12, cmd, col)
        screen.blit(f_small.render(f"{cmd:+5.2f}", True, col), (x0 + 160, y + 1))
        screen.blit(f_small.render(f"{deg:+6.1f}d", True, DIM), (x0 + 208, y + 1))
        y += 15
    thr, pos = c["throttle_cmd"], c["throttle_pos"]
    screen.blit(f_small.render("throttle", True, DIM), (x0, y + 2))
    capped = (not stick) and thr >= throttle_cap - 1e-6
    _bar(screen, x0 + 58, y, 96, 12, thr, WARN if capped or pos > 1.0 else GOOD)
    # solid line where the cap bites, faint where it is only drawn for reference
    pygame.draw.line(screen, DIM if stick else FG,
                     (x0 + 58 + int(96 * throttle_cap), y - 2),
                     (x0 + 58 + int(96 * throttle_cap), y + 14), 1 if stick else 2)
    pygame.draw.line(screen, DIM, (x0 + 58 + 48, y + 12), (x0 + 58 + 48, y + 16), 1)
    screen.blit(f_small.render(f"{thr:5.2f}", True, FG), (x0 + 160, y + 1))
    screen.blit(f_small.render(f"pos {pos:.2f}{' AB' if pos > 1.0 else ''}", True,
                               WARN if pos > 1.0 else DIM), (x0 + 200, y + 1))
    y += 17
    line(f"| cap {throttle_cap:.2f}" + ("  (not enforced in STICK)" if stick
                                        else "  D23 balance knob, not physics"),
         DIM, f_small, 16)

    y = H - 102
    # Same keys in both layers, one exception: W/8 is stick-forward here and a
    # climb over there.  Name the direction so tab cannot catch you out.
    if stick:
        line("W/8 NOSE DOWN   X/2 nose up   A D roll", WARN, f_small, 14)
        line(f"Q E rudder   Z C throttle   S centre   ramp {ramp_s:.2f}s",
             DIM, f_small, 14)
    else:
        line("W/8 CLIMB   X/2 descend   A D heading", GOOD, f_small, 14)
        line("Z C speed   S freeze targets   m CONTINUOUS/DISCRETE", DIM, f_small, 14)
        line("shift = coarse (10 deg / 100 ft / 20 kt)", DIM, f_small, 14)
    line("tab layer   p pause   r reset   esc quit", DIM, f_small, 14)
    line(f"+ - zoom ({scale:.0f} m/px)   g trail", DIM, f_small, 14)
    rt_col = GOOD if rtf > 0.92 else (WARN if rtf > 0.7 else BAD)
    line(f"t = {s.t:6.1f} s   real time x{rtf:.2f}"
         + (f"   tacview {'up' if live.connected else 'waiting'}" if live else ""),
         rt_col, f_small, 14)

    if crash is not None:
        box = pygame.Rect(MAP_W // 2 - 210, 26, 420, 58)
        pygame.draw.rect(screen, (60, 20, 24), box)
        pygame.draw.rect(screen, BAD, box, 2)
        screen.blit(f_big.render("CRASHED   r to reset", True, BAD),
                    (box.x + 16, box.y + 8))
        screen.blit(f_small.render(
            f"t {crash[0]:.1f} s   {crash[1]:.0f} kt   gamma {crash[2]:+.0f} deg"
            f"   {crash[3]:.0f} ft", True, FG), (box.x + 16, box.y + 36))
    elif handover is not None:
        screen.blit(f_big.render(f"-> {handover[1]}", True, GOOD),
                    (MAP_W // 2 - 60, 26))
    elif s.h_ft < 2000:
        screen.blit(f_big.render("PULL UP", True, BAD), (MAP_W // 2 - 44, 30))
    if paused:
        screen.blit(f_big.render("PAUSED", True, WARN), (MAP_W // 2 - 44, 62))


def _ray(screen, cx, cy, ang, length, col, width):
    pygame.draw.line(screen, col, (cx, cy),
                     (cx + math.sin(ang) * length, cy - math.cos(ang) * length), width)


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
