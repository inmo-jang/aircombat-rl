"""Fly a task with a policy instead of a keyboard, and watch it.

    from aircombat_gym.tools.auto_operation import fly
    fly(env, act)          # act(obs) -> action

The sibling of `manual_operation`: same three views, same layout, same
`render.py`.  The only difference is where the actions come from -- a keyboard
there, an `act(obs)` here -- and what the readout has room to say.

`act` is anything that turns an observation into an action, so this plays a
trained policy, a scripted bot or a constant.  It drives the *task env*, which
means what you are watching is what gets graded; a viewer built on its own
simulation is a nice picture of the wrong fight.

    BACKSPACE   throw this episode away, draw a new random one
    SPACE       pause
    TAB         1x, 2x, 4x, uncapped
    ESC / Q     quit

TacView Advanced (Windows only) can take the same fight live over TCP; pass
`sinks=[AcmiRealtime()]`.  Everyone else gets the pygame views, which is why
they exist.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from . import render
from .render import (ACCENT, BAD, BG, DIM, FG, GOOD, WARN, Layout, Track,
                     Weapon, bar, gap, head, text)

SPEEDS = (1.0, 2.0, 4.0, 0.0)          # 0 = uncapped
TRAIL_MAX = 2400                       # one full episode at 20 Hz


def fly(env, act, *, title="policy", subtitle="", seed=None, sinks=(),
        on_episode=None):
    """Run episodes until the window closes.  Returns (kills, episodes).

    `sinks` are TacView writers (`core.tacview`); they get a monotonic clock
    across episodes, because resetting it rewinds the recording and collapses
    every episode onto one timeline.
    """
    sc, fonts, clock = render.open_window(title)
    f_big, f, fs = fonts
    readout = render.Readout(sc, fonts)

    weapon = Weapon.from_env(env)
    duel = env._duel
    rng = np.random.default_rng(seed)
    own, foe = Track(None), Track(None)
    scale = 12.0
    ep, kills, finished = 0, 0, 0
    speed_i, paused, wall = 0, False, 0.0

    def refresh():
        own.state = duel.ac["red"].state
        foe.state = duel.ac["blue"].state

    def new_episode():
        nonlocal ep
        ep += 1
        s = int(rng.integers(0, 2**31 - 1))
        obs, info = env.reset(seed=s)
        own.trail.clear(); own.profile.clear(); own.dead = False
        foe.trail.clear(); foe.profile.clear(); foe.dead = False
        refresh()
        render.emit_event(sinks, f"episode {ep} seed {s}")
        return obs, info, s

    obs, info, seed_used = new_episode()
    outcome, running = "", True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_BACKSPACE:
                    obs, info, seed_used = new_episode()
                    outcome = ""
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_TAB:
                    speed_i = (speed_i + 1) % len(SPEEDS)

        if not paused and not outcome:
            obs, _, term, trunc, info = env.step(act(obs))
            wall += 1.0 / 20.0
            refresh()
            render.push(own, info["t"], TRAIL_MAX)
            render.push(foe, info["t"], TRAIL_MAX)
            foe.dead = info.get("opp_health", 1.0) <= 0.0
            render.emit_frame(sinks, wall, own.state, foe.state,
                              locked=bool(info.get("in_wez")))
            if term or trunc:
                won = bool(info.get("won"))
                outcome = "KILL" if won else "TIMEOUT"
                kills += won
                finished += 1
                render.emit_event(
                    sinks, f"episode {ep}: {outcome} at {info['t']:.0f}s")
                if on_episode:
                    on_episode(ep, won, info)

        scale = render.fit_scale(scale, info["range"])

        # accept a viewer even while paused or between episodes
        render.poll_sinks(sinks)

        in_wez = bool(info.get("in_wez"))
        lock = min(1.0, info.get("track_time", 0.0) / weapon.lock_s)
        aim = math.degrees(info.get("ata_lead", 0.0))
        hp = info.get("opp_health", 1.0)

        render.world_views(sc, own, foe, weapon, scale=scale, in_wez=in_wez,
                           lock_frac=lock, profile_autoscale=True, fs_font=fs)
        render.cockpit(sc, Layout.cockpit, (f, fs), own, foe, weapon,
                       in_wez=in_wez, lock_frac=lock)
        readout.draw([
            text(f"episode {ep}   {outcome}" if outcome
                 else f"episode {ep}   t {info.get('t', 0.0):5.1f}s",
                 FG, big=True),
            text(subtitle, DIM, small=True),
            gap(10),
            text(f"score  {kills}/{finished}"
                 + (f"   ({kills/finished:.0%})" if finished else ""), FG),
            text(f"seed   {seed_used}", DIM, small=True),
            gap(6),
            head("GEOMETRY"),
            text(f"range     {info.get('range', 0):7.0f} m", FG),
            text(f"aim error {aim:7.1f} deg",
                 GOOD if aim <= weapon.cone_deg else FG),
            text(f"aspect    {math.degrees(info.get('aa', 0)):7.1f} deg", FG),
            text(f"closure   {-info.get('range_rate', 0):7.0f} m/s", FG),
            text(f"own speed {info.get('own_speed', 0):7.0f} kt", FG),
            gap(6),
            head("SHOT"),
            text(f"in WEZ    {'YES' if in_wez else 'no'}",
                 GOOD if in_wez else DIM),
            bar(f"lock (of {weapon.lock_s:.1f}s)", lock,
                GOOD if lock >= 1.0 else WARN, f"{lock * weapon.lock_s:.2f}s"),
            bar("target health", hp, BAD if hp < 1.0 else DIM, f"{hp:.2f}"),
        ], pinned=[
            text("BACKSPACE new episode   SPACE pause", DIM, small=True),
            text(f"TAB replay x{'max' if not SPEEDS[speed_i] else SPEEDS[speed_i]:g}"
                 f"   ESC quit", DIM, small=True),
        ])
        pygame.display.flip()
        clock.tick(20.0 * SPEEDS[speed_i] if SPEEDS[speed_i] else 0)

    pygame.quit()
    return kills, finished


