"""Be the policy.  Fly a task by hand through the surface a policy trains on.

    python -m aircombat_gym.wvr.play --env circular
    python -m aircombat_gym.wvr.play --env advantaged --continuous --tacview

Every keypress becomes an element of `env.action_space` and goes through
`env.step()`.  Nothing here reaches around the environment: same weapon, same
clock, same opponent, same `info`.  What you feel is what a policy is up
against, which is the point -- a reward written without having flown the task is
a guess about a game you have not played.

**This is not `tools/manual_operation`.**  That one is for feeling the
*aircraft*: it has a control-surface layer, a fine setpoint mode, and a panel
full of energy and alpha, and most of what it can do a policy cannot ask for.
This one is deliberately poorer.  Four arrow keys, because the action space has
two open axes and three values each, and a readout of the material a reward is
built from and nothing else.

Holding a key repeats the action, which is not a shortcut: a policy also emits
one action per decision step, and holding is what a sustained manoeuvre looks
like from inside the action space.  A zero delta freezes the target rather than
re-deriving it, so letting go is itself a command.

    arrows      heading -/+ , speed -/+     (nothing held = the no-op action)
    r           new engagement
    p           pause
    m           discrete <-> continuous, mid-episode
    + -  g      zoom, trail length
    esc         quit

`--continuous` swaps `Discrete(9)` for `Box(2)`: the same two axes, but the key
ramps the demand up while it is held instead of slamming it to the limit.  Fly
both -- the choice of action space moves a score as much as the reward does.
"""
from __future__ import annotations

import argparse
import math
import time

# Nothing above this line imports a viewer, and `wvr/__init__` does not import
# this module, so `import aircombat_gym.wvr` still costs no pygame.  The window
# is a property of running this file, not of the package.
from .envs import ENVS          # importing this registers the gym ids

# Short names to type, spelled out rather than derived: lowercasing the id
# gives `advantagedfight`, which nobody wants at a prompt.  The assertion is
# what keeps the list honest -- add an environment without naming it here and
# this module refuses to load rather than quietly dropping it from `--env`.
ENV_BY_NAME = {"circular": "AirCombat/Circular-v0",
               "evader": "AirCombat/Evader-v0",
               "advantaged": "AirCombat/AdvantagedFight-v0",
               "fair": "AirCombat/FairFight-v0"}
assert set(ENV_BY_NAME.values()) == set(ENVS), (
    f"--env is missing {set(ENVS) - set(ENV_BY_NAME.values())}")
KEY_REPEAT_MS = (220, 50)
TRAIL_LENGTHS = (600, 2400, 0)          # 0 = unlimited


def _held(keys, pygame, neg, pos) -> float:
    """-1, 0 or +1 from a pair of key groups.  Both down cancels."""
    return ((1.0 if any(keys[k] for k in pos) else 0.0)
            - (1.0 if any(keys[k] for k in neg) else 0.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="circular", choices=sorted(ENV_BY_NAME),
                    help="which task to fly")
    ap.add_argument("--continuous", action="store_true",
                    help="Box(2) instead of Discrete(9)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--scale", type=float, default=8.0, help="metres per pixel")
    ap.add_argument("--ramp", type=float, default=0.35,
                    help="seconds of held key to full demand, --continuous only")
    ap.add_argument("--tacview", action="store_true")
    ap.add_argument("--tacview-port", type=int, default=42674)
    ap.add_argument("--tacview-wait", type=float, default=0.0)
    ap.add_argument("--acmi", metavar="PATH")
    args = ap.parse_args(argv)

    import gymnasium as gym
    import pygame

    from ..core.aircraft import DECISION_HZ
    from ..core.tacview import AcmiFile, AcmiRealtime
    from ..tools import render
    from . import actions as act

    env = gym.make(ENV_BY_NAME[args.env],
                   action_mode="continuous" if args.continuous else "discrete")
    task = env.unwrapped
    combat = task._combat

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
            print("  connected" if live.wait(args.tacview_wait)
                  else "  no connection yet; flying anyway")
        sinks.append(live)

    pygame.init()
    screen, fonts, clock = render.open_window(f"be the policy - {args.env}")
    pygame.key.set_repeat(*KEY_REPEAT_MS)
    weapon = render.Weapon.from_env(env)

    LAT = (pygame.K_LEFT,), (pygame.K_RIGHT,)
    LON = (pygame.K_DOWN,), (pygame.K_UP,)

    obs, info = env.reset(seed=args.seed)
    own = render.Track(combat.ac[task.seat].state)
    foe = render.Track(combat.ac[task.foe_seat].state)
    demand = [0.0, 0.0]                 # continuous only: where the ramp is now
    continuous = args.continuous
    n_steps, outcome, paused = 0, None, False
    trail_i, scale = 1, args.scale
    dt = 1.0 / DECISION_HZ
    accum, last_wall = 0.0, time.perf_counter()
    action = task.encode_action((0.0, 0.0, 0.0))

    def restart():
        nonlocal obs, info, own, foe, n_steps, outcome, accum
        obs, info = env.reset()
        own = render.Track(combat.ac[task.seat].state)
        foe = render.Track(combat.ac[task.foe_seat].state)
        demand[:] = [0.0, 0.0]
        n_steps, outcome, accum = 0, None, 0.0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    restart()
                elif e.key == pygame.K_p:
                    paused = not paused
                elif e.key == pygame.K_m:
                    # The episode carries on -- only the space changes, so the
                    # same fight can be flown both ways.  The ramp is zeroed so
                    # the first continuous step does not inherit a demand the
                    # discrete mode never had.
                    continuous = not continuous
                    task.set_action_mode("continuous" if continuous
                                         else "discrete")
                    demand[:] = [0.0, 0.0]
                elif e.key == pygame.K_g:
                    trail_i = (trail_i + 1) % len(TRAIL_LENGTHS)
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    scale = max(2.0, scale / 1.3)
                elif e.key == pygame.K_MINUS:
                    scale = min(60.0, scale * 1.3)

        now = time.perf_counter()
        frame_dt = now - last_wall
        last_wall = now

        keys = pygame.key.get_pressed()
        want = (_held(keys, pygame, *LAT), _held(keys, pygame, *LON))
        if continuous:
            # Hold to lean on it.  Slamming to the limit on the first frame
            # would make Box(2) a three-valued grid with extra steps, which is
            # exactly the difference the two modes exist to show.
            #
            # On the wall clock, not per frame.  It was `dt / ramp` -- the
            # decision period over the ramp time -- added once a *frame*, so at
            # 60 fps the demand saturated in 0.12 s instead of 0.35 and a short
            # tap landed on whatever fraction the frame boundary happened to
            # give it.  That reads as the keys moving at random, because the
            # same press twice gave two different turns.
            rate = frame_dt / max(args.ramp, 1e-3)
            for i in (0, 1):
                if want[i]:
                    demand[i] = max(-1.0, min(1.0, demand[i] + want[i] * rate))
                else:
                    demand[i] -= math.copysign(min(abs(demand[i]), rate * 2.0),
                                               demand[i])
            action = task.encode_action((demand[0] * act.DELTA_HEADING_DEG[-1],
                                         demand[1] * act.DELTA_SPEED_KT[-1], 0.0))
        else:
            action = task.encode_action((want[0] * act.DELTA_HEADING_DEG[-1],
                                         want[1] * act.DELTA_SPEED_KT[-1], 0.0))

        accum += frame_dt
        if paused or outcome is not None:
            accum = 0.0
        n_this_frame = 0
        while accum >= dt and n_this_frame < 4 and outcome is None:
            accum -= dt
            n_this_frame += 1
            obs, _r, term, trunc, info = env.step(action)
            n_steps += 1
            own.state = combat.ac[task.seat].state
            foe.state = combat.ac[task.foe_seat].state
            foe.dead = combat.health[task.foe_seat] <= 0.0
            cap = TRAIL_LENGTHS[trail_i]
            render.push(own, own.state.t, cap)
            render.push(foe, foe.state.t, cap)
            render.emit_frame(sinks, own.state.t, own.state, foe.state,
                              locked=info["in_wez"])
            if term or trunc:
                outcome = info.get("outcome", "?")
                render.emit_event(sinks, outcome.upper())

        # The same speed vector and heading marker the keyboard tool draws, off
        # the same two attributes.  Worth having here even though this tool is
        # deliberately spare: an action is a *delta* on these, so the bar is
        # where you see what your keypress actually asked for -- the nose swings
        # to the marker, and the bar shows the speed the last press committed to
        # rather than the speed you have.
        mine = combat.ac[task.seat]
        scale = render.fit_scale(scale, info["range"])
        render.world_views(screen, own, foe, weapon, scale=scale,
                           in_wez=info["in_wez"],
                           lock_frac=min(1.0, info["track_time"] / weapon.lock_s),
                           autopilot=dict(psi_cmd=mine.psi_cmd,
                                         v_cmd_kt=mine.v_cmd_kt),
                           alt_cmd_ft=mine.alt_cmd_ft,
                           # his cone too, in red.  Every task here arms both
                           # aircraft, so leaving it out would draw half a game
                           foe_wez=(bool(info["opp_in_wez"]),
                                    info["under_track"] >= weapon.lock_s),
                           fs_font=fonts[2])
        render.cockpit(screen, render.Layout.cockpit, (fonts[1], fonts[2]),
                       own, foe, weapon, in_wez=info["in_wez"],
                       lock_frac=min(1.0, info["track_time"] / weapon.lock_s))
        _readout(render, screen, fonts, env, task, info, action, n_steps,
                 outcome, paused, continuous, live, demand)
        render.present(screen)
        render.poll_sinks(sinks)
        clock.tick(60)

    for k in sinks:
        k.close()
    env.close()
    pygame.quit()
    return 0


def _readout(render, screen, fonts, env, task, info, action, n_steps,
             outcome, paused, continuous, live, demand):
    """Only what a policy can see, plus what a reward would be built from.

    No energy bar, no `Ps`, no alpha limiter.  Those belong to the tool for
    feeling the aircraft; here they would show the pilot something the policy
    is not given and quietly change what the exercise teaches.
    """
    T, head, gap, bar = render.text, render.head, render.gap, render.bar
    d = task.decode(action)
    space = env.action_space          # live: `m` swaps it mid-episode
    label = (f"Box({space.shape[0]})" if continuous
             else f"Discrete({space.n})")

    items = [
        T(f"{type(task).__name__.replace('Env', '')}   {label}",
          render.ACCENT, big=True),
        T(f"step {n_steps:5d}    t {info['t']:6.2f} s", render.DIM, small=True),
        gap(),
        head("ACTION  -- what you just sent"),
        T(f"  {'' if continuous else f'index {int(action)}  '}"
          f"dpsi {d[0]:+6.1f} deg   dV {d[1]:+5.1f} kt",
          render.WARN if any(d) else render.DIM),
    ]
    if continuous:
        # The ramp is the whole of continuous mode, so show where it is rather
        # than leaving the pilot to infer it from the aircraft's response.
        items += [bar("  hdg demand", (demand[0] + 1) / 2, render.ACCENT,
                      f"{demand[0]:+.2f}", width=150),
                  bar("  spd demand", (demand[1] + 1) / 2, render.ACCENT,
                      f"{demand[1]:+.2f}", width=150)]
    items += [
        gap(),
        head("INFO  -- the material a reward is built from"),
        T(f"  range      {info['range']:8.0f} m  closing {-info['range_rate']:+6.1f}"),
        T(f"  ata        {math.degrees(info['ata']):8.1f} deg  lead"
          f" {math.degrees(info['ata_lead']):5.1f}"),
        T(f"  aa         {math.degrees(info['aa']):8.1f} deg"),
        T(f"  in_wez     {str(bool(info['in_wez'])):>8}"
          f"  opp {str(bool(info['opp_in_wez'])):>5}",
          render.GOOD if info["in_wez"] else render.FG),
        T(f"  track_time {info['track_time']:8.2f} s  under {info['under_track']:.2f}"),
        gap(4),
        bar("own_health", info["own_health"], render.ACCENT,
            f"{info['own_health']:.2f}"),
        bar("opp_health", info["opp_health"], render.BAD,
            f"{info['opp_health']:.2f}"),
    ]
    if outcome is not None:
        items += [gap(), T(f"{outcome.upper()}   r for a new engagement",
                           render.GOOD if outcome == "kill" else render.WARN,
                           big=True)]
    elif paused:
        items += [gap(), T("PAUSED", render.WARN, big=True)]

    pinned = [
        T("arrows  heading -/+   speed -/+   none = no-op",
          render.DIM, small=True),
        T("r reset  p pause  m discrete/continuous  esc quit",
          render.DIM, small=True),
        T("+ - zoom   g trail", render.DIM, small=True),
        T("every keypress goes through env.step()"
          + ("   tacview up" if live is not None and live.connected else ""),
          render.DIM, small=True),
    ]
    render.Readout(screen, fonts, render.Layout.readout).draw(items, pinned)


if __name__ == "__main__":
    raise SystemExit(main())
