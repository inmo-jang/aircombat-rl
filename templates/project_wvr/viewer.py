"""A match on screen, with neither aircraft as "own".

    from viewer import watch
    watch(env, red, blue, plan, speed=1.0, sinks=[], on_result=record)

`tools.watch` shows one policy against an opponent, from that policy's seat:
the view follows it, the panel is its instruments, TacView calls it Blue.  A
match has two entrants and owes them the same picture, so this draws

    a header        the engagement, the result as it lands, the replay speed
    two panels      the same instruments for each team, red left, blue right,
                    each with its altitude against time
    the plan view   full width, centred between the two aircraft, both in
                    their team's colour, both trails, both firing cones, the
                    range between them -- and the score, big, along the bottom

and sends TacView each team under its own name and colour.  Engagements follow
one another on their own; BACKSPACE moves on at once.

    SPACE pause   BACKSPACE next   TAB 1x/2x/4x/max   + - twice/half   ESC quit
"""
from __future__ import annotations

import math
import time

import pygame

from aircombat_gym.core.tacview import state_to_object
from aircombat_gym.tools import render
from aircombat_gym.tools.render import (BAD, BG, DIM, FG, GOOD, GRID, PANEL, WARN,
                                        Track, Weapon, bar, gap, text)

TEAM = {   # body, trail, cone (idle), cone (in WEZ), TacView colour and 3D model
    # The model is the look in TacView only -- both fly the same F-16.
    "red": dict(body=(240, 90, 90), trail=(130, 58, 58), idle=(92, 52, 52),
                hot=(255, 140, 140), acmi="Red", acmi_id="101", model="MiG-29A"),
    "blue": dict(body=(90, 160, 255), trail=(52, 88, 140), idle=(48, 66, 100),
                 hot=(150, 200, 255), acmi="Blue", acmi_id="102", model="F-16C"),
}
SPEEDS = (1.0, 2.0, 4.0, 0.0)          # TAB cycles these.  0 = uncapped
SPEED_MIN, SPEED_MAX = 0.125, 64.0
HOLD_S = 2.0                           # the result stays this long at 1x
RESULT = {"kill": "red", "died": "blue"}
ALT_REF_FT = 20_000.0                  # the altitude both start at, and are held to


def _cone(sc, cx, cy, psi, weapon: Weapon, scale, col):
    """Horizontal slice of one aircraft's firing cone."""
    r_in, r_out = weapon.r_min / scale, weapon.r_max / scale
    half = math.radians(weapon.cone_deg)
    for side in (-1, 1):
        a = psi + side * half
        pygame.draw.line(sc, col, (cx + math.sin(a) * r_in, cy - math.cos(a) * r_in),
                         (cx + math.sin(a) * r_out, cy - math.cos(a) * r_out), 1)
    for rr in (r_in, r_out):
        if rr > 3:
            pygame.draw.arc(sc, col, pygame.Rect(cx - rr, cy - rr, 2 * rr, 2 * rr),
                            math.pi / 2 - psi - half, math.pi / 2 - psi + half, 1)


def _plan(sc, rect, font, tracks, snaps, weapon, scale):
    """Both aircraft, centred between them."""
    x0, y0, w, h = rect
    sc.fill(BG, rect)
    old = sc.get_clip()
    sc.set_clip(pygame.Rect(rect))
    a, b = tracks["red"].state, tracks["blue"].state
    mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
    cx, cy = x0 + w // 2, y0 + h // 2

    def px(x, y):
        return int(cx + (x - mx) / scale), int(cy - (y - my) / scale)

    step = 1000.0 / scale
    if step >= 12:
        gx = cx + (-mx / scale) % step - step * math.ceil(w / step)
        while gx < x0 + w:
            pygame.draw.line(sc, GRID, (int(gx), y0), (int(gx), y0 + h))
            gx += step
        gy = cy + (my / scale) % step - step * math.ceil(h / step)
        while gy < y0 + h:
            pygame.draw.line(sc, GRID, (x0, int(gy)), (x0 + w, int(gy)))
            gy += step

    for team in ("red", "blue"):
        tr = tracks[team].trail
        if len(tr) > 1:
            stride = max(1, len(tr) // 400)
            pts = [px(x, y) for x, y in tr[::stride]]
            if len(pts) > 1:
                pygame.draw.lines(sc, TEAM[team]["trail"], False, pts, 2)

    (ax, ay), (bx, by) = px(a.x, a.y), px(b.x, b.y)
    pygame.draw.line(sc, (70, 76, 92), (ax, ay), (bx, by), 1)
    rng = math.dist((a.x, a.y, a.h), (b.x, b.y, b.h))
    sc.blit(font.render(f"{rng:,.0f} m", True, DIM), ((ax + bx) // 2 + 8, (ay + by) // 2))

    for team, other in (("red", "blue"), ("blue", "red")):
        s, snap = tracks[team].state, snaps[team]
        x, y = px(s.x, s.y)
        locked = snap["in_wez"] and snap["track_time"] >= weapon.lock_s
        _cone(sc, x, y, s.psi, weapon, scale,
              TEAM[team]["hot"] if snap["in_wez"] else TEAM[team]["idle"])
        if locked and not tracks[other].dead:
            o = tracks[other].state
            pygame.draw.circle(sc, TEAM[team]["hot"], px(o.x, o.y), 24, 2)
    for team in ("red", "blue"):
        s = tracks[team].state
        col = (120, 120, 120) if tracks[team].dead else TEAM[team]["body"]
        render.plane(sc, *px(s.x, s.y), s.psi, col, 18)

    sc.blit(font.render(f"{1000.0 / scale:.0f} px = 1 km", True, DIM),
            (x0 + 12, y0 + 8))
    sc.set_clip(old)


def _scoreboard(sc, rect, fonts, names, score):
    """The score, big, at the bottom centre of the plan view."""
    f_team, f_num, f_small = fonts
    x0, y0, w, h = rect
    name_w = max(80, int(w * 0.3))
    parts = [f_team.render(_fit(f_team, names["red"], name_w), True, TEAM["red"]["body"]),
             f_num.render(f"  {score['red']} - {score['blue']}  ", True, FG),
             f_team.render(_fit(f_team, names["blue"], name_w), True, TEAM["blue"]["body"])]
    rest = f_small.render(f"mutual {score['mutual']}    timeout {score['timeout']}",
                          True, DIM)
    row_w = sum(p.get_width() for p in parts)
    row_h = max(p.get_height() for p in parts)
    box_w, box_h = max(row_w, rest.get_width()) + 40, row_h + rest.get_height() + 22
    bx, by = x0 + (w - box_w) // 2, y0 + h - box_h - 10
    box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    box.fill((*PANEL, 225))
    sc.blit(box, (bx, by))
    x = bx + (box_w - row_w) // 2
    for p in parts:
        sc.blit(p, (x, by + 8 + (row_h - p.get_height()) // 2))
        x += p.get_width()
    sc.blit(rest, (bx + (box_w - rest.get_width()) // 2, by + 10 + row_h))
    return bx, by, box_w, box_h, by + 8 + row_h // 2


def _health_bars(sc, map_rect, box, font, health, ghost):
    """A fighting game's life bars, either side of the score.

    Each bar hangs from its outer end and empties towards the score, so what
    is left sits at the edge of the screen.  `ghost` is the health a moment
    ago: lost health shows as a pale stretch that then catches up.
    """
    x0, y0, w, h = map_rect
    bx, by, box_w, box_h, mid_y = box
    bar_h, pad = 26, 14
    if (w - box_w) // 2 - 2 * pad >= 120:               # beside the score
        spans = {"red": (x0 + pad, bx - pad), "blue": (bx + box_w + pad, x0 + w - pad)}
        y = mid_y - bar_h // 2
    else:                                                # no room: above it
        spans = {"red": (x0 + pad, x0 + w // 2 - 8), "blue": (x0 + w // 2 + 8, x0 + w - pad)}
        y = by - bar_h - 10
    for team, (left, right) in spans.items():
        width = right - left
        pygame.draw.rect(sc, (36, 40, 50), (left, y, width, bar_h))
        hp, gh = max(0.0, health[team]), max(0.0, ghost[team])
        if team == "red":           # hangs from the left, empties to the right
            g = (left, y, int(width * gh), bar_h)
            f = (left, y, int(width * hp), bar_h)
        else:                       # hangs from the right, empties to the left
            g = (right - int(width * gh), y, int(width * gh), bar_h)
            f = (right - int(width * hp), y, int(width * hp), bar_h)
        pygame.draw.rect(sc, (235, 225, 200), g)
        pygame.draw.rect(sc, TEAM[team]["body"], f)
        pygame.draw.rect(sc, (200, 205, 215), (left, y, width, bar_h), 2)
        label = font.render("K.O." if hp <= 0.0 else f"{hp:.0%}", True, FG)
        lx = left + 8 if team == "red" else right - label.get_width() - 8
        sc.blit(label, (lx, y + (bar_h - label.get_height()) // 2))


def _left(dq: str, names: dict) -> str:
    """`info["disqualified"]` -> who left the altitude band, in words."""
    who = "both" if dq == "both" else names[dq]
    return f"{who} left the altitude band"


def _fit(font, s: str, width: int) -> str:
    """`s`, cut to `width` pixels."""
    if font.size(s)[0] <= width:
        return s
    while s and font.size(s + "...")[0] > width:
        s = s[:-1]
    return s + "..."


def _altitude(sc, rect, font, profile, col, t_max, band):
    """Altitude against time, with the reference and the band (if any) marked."""
    x0, y0, w, h = rect
    pygame.draw.rect(sc, (20, 23, 30), rect)
    alts = [a for _, a in profile]
    span = 1.5 * (band or 1000.0)
    lo, hi = ALT_REF_FT - span, ALT_REF_FT + span
    if alts:
        lo, hi = min(lo, min(alts) - 200.0), max(hi, max(alts) + 200.0)
    top, bot = y0 + 20, y0 + h - 6

    def ypx(ft):
        return int(bot - (ft - lo) / (hi - lo) * (bot - top))

    lines = [(ALT_REF_FT, (60, 66, 80))]
    if band:
        lines += [(ALT_REF_FT + band, (150, 70, 60)), (ALT_REF_FT - band, (150, 70, 60))]
    for ft, c in lines:
        y = ypx(ft)
        pygame.draw.line(sc, c, (x0 + 34, y), (x0 + w - 4, y), 1)
        sc.blit(font.render(f"{ft / 1000:.0f}k", True, DIM), (x0 + 4, y - 7))
    if len(profile) > 1:
        t0 = profile[0][0]
        pts = [(int(x0 + 34 + (t - t0) / t_max * (w - 38)), ypx(a)) for t, a in profile]
        pygame.draw.lines(sc, col, False, pts, 2)
    if alts:
        # out of the band: orange and marked, so it reads on the red team too
        out = bool(band) and abs(alts[-1] - ALT_REF_FT) > band
        label = f"{alts[-1]:,.0f} ft" + ("  OUT" if out else "")
        if font.size("altitude " + label)[0] < w - 8:
            label = "altitude " + label
        sc.blit(font.render(label, True, WARN if out else col), (x0 + 4, y0 + 3))


def _panel(readout, team, name, note, start, snap, weapon, width, text_w):
    """One team's instruments.  The same lines for both."""
    col = TEAM[team]["body"]
    hp = snap["own_health"]
    lock = min(1.0, snap["track_time"] / weapon.lock_s)
    aim = math.degrees(snap["ata_lead"])
    readout.draw([
        text(_fit(readout.f_big, name, text_w), col, big=True),
        text(_fit(readout.fs, note, text_w), DIM, small=True),
        text(f"from {start}", DIM, small=True),
        gap(14),
        bar("health", hp, BAD if hp < 0.5 else (WARN if hp < 1.0 else GOOD),
            f"{hp:.2f}", width=width),
        bar(f"lock (of {weapon.lock_s:.1f}s)", lock, GOOD if lock >= 1.0 else WARN,
            f"{lock * weapon.lock_s:.2f}s", width=width),
        gap(8),
        text(f"in WEZ     {'YES' if snap['in_wez'] else 'no'}",
             GOOD if snap["in_wez"] else DIM),
        text(f"aim error  {aim:5.1f} deg", GOOD if aim <= weapon.cone_deg else FG),
        text(f"speed      {snap['own_speed']:5.0f} kt", FG),
        text(f"under fire {'YES' if snap['opp_in_wez'] else 'no'}",
             BAD if snap["opp_in_wez"] else DIM),
    ])


def watch(env, red, blue, plan, *, speed=1.0, sinks=(), on_result=None):
    """Fly the engagements `plan` yields until it runs out or the window closes.

    `env` is an `ArenaEnv` with `blue` as its opponent; `red` flies its red
    seat throughout, so each team keeps its colour while the starts swap.
    `plan` yields `(seed, swapped)`.  `on_result(seed, swapped, info)` is called
    for every engagement that runs to the end.
    """
    duel = env._combat
    weapon = Weapon.from_env(env)
    sc, fonts, clock = render.open_window(f"project_wvr  -  {red.name} vs {blue.name}")
    f_big, f, fs = fonts
    f_speed = pygame.font.SysFont("consolas,menlo,monospace", 40, bold=True)
    W, H = sc.get_size()
    f_score = pygame.font.SysFont("consolas,menlo,monospace", 52, bold=True)
    f_team = pygame.font.SysFont("consolas,menlo,monospace", 30, bold=True)
    head_h, panel_h, foot_h = 76, 232, 24
    half = W // 2
    side = {"red": render.Readout(sc, fonts, (0, head_h, half, panel_h)),
            "blue": render.Readout(sc, fonts, (half, head_h, W - half, panel_h))}
    map_rect = (0, head_h + panel_h, W, H - head_h - panel_h - foot_h)
    # each panel: instruments on the left, the altitude chart on the right
    text_col = max(int(half * 0.56), min(232, half - 80))  # the text needs ~230 px
    text_w = text_col - 2 * render.Readout.PAD_X
    bar_w = min(200, text_w - 64)
    chart = {"red": (text_col, head_h + 12, half - text_col - 14, panel_h - 24),
             "blue": (half + text_col, head_h + 12, W - half - text_col - 14,
                      panel_h - 24)}

    tracks = {"red": Track(None), "blue": Track(None)}
    score = {"red": 0, "blue": 0, "mutual": 0, "timeout": 0}
    names = {"red": red.name, "blue": blue.name}
    notes = {"red": red.note, "blue": blue.note}
    speed = float(speed)
    paused, running, done = False, True, False
    scale, wall = 12.0, 0.0
    k, seed, swapped, obs, outcome, ended_at = 0, None, False, None, None, 0.0
    dq = None                          # who left the altitude band, if anyone
    band = getattr(env, "altitude_band_ft", 0.0)
    ghost = {"red": 1.0, "blue": 1.0}  # the life bars' recent health
    last_hp = {"red": 1.0, "blue": 1.0}
    wait = {"red": 0, "blue": 0}       # frames the pale stretch still waits
    plan = iter(plan)

    def start_next() -> bool:
        nonlocal k, seed, swapped, obs, outcome, dq
        for d in (ghost, last_hp):
            d.update(red=1.0, blue=1.0)
        wait.update(red=0, blue=0)
        try:
            seed, swapped = next(plan)
        except StopIteration:
            return False
        k += 1
        obs, _ = env.reset(seed=seed, options={"swapped": swapped})
        for team in ("red", "blue"):
            tracks[team].trail.clear(); tracks[team].profile.clear()
            tracks[team].dead = False
            tracks[team].state = duel.ac[team].state
        outcome, dq = None, None
        render.emit_event(sinks, f"engagement {k}: seed {seed}, "
                                 f"{'swapped' if swapped else 'as configured'}")
        return True

    done = not start_next()
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_BACKSPACE and not done:
                    done = not start_next()
                elif e.key == pygame.K_TAB:
                    speed = (next((s for s in SPEEDS if s > speed), 0.0)
                             if speed else SPEEDS[0])
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = 0.0 if not speed or speed * 2 > SPEED_MAX else speed * 2
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = SPEED_MAX if not speed else max(speed / 2, SPEED_MIN)

        if not done and not paused and outcome is None:
            obs, _, term, trunc, info = env.step(red.act(obs))
            wall += 1.0 / 20.0
            for team in ("red", "blue"):
                tracks[team].state = duel.ac[team].state
                render.push(tracks[team], info["t"])
                tracks[team].dead = duel.health[team] <= 0.0
            snaps = env._last
            objs = []
            for team, other in (("red", "blue"), ("blue", "red")):
                # TacView picks the 3D model by Name, so Name is the type
                o = state_to_object(tracks[team].state, TEAM[team]["acmi_id"],
                                    TEAM[team]["model"], TEAM[team]["acmi"])
                o["callsign"] = o["pilot"] = names[team]   # the label beside it
                o["locked"] = TEAM[other]["acmi_id"] if snaps[team]["in_wez"] else None
                objs.append(o)
            for s in sinks:
                s.frame(wall, objs)
            if term or trunc:
                outcome, ended_at = info["outcome"], time.perf_counter()
                dq = info.get("disqualified")
                score[RESULT.get(outcome, outcome)] += 1
                winner = RESULT.get(outcome)
                note = f" ({_left(dq, names)})" if dq else ""
                render.emit_event(sinks, (f"{names[winner]} wins" if winner
                                          else outcome.upper()) + note)
                if on_result:
                    on_result(seed, swapped, info)

        hold = HOLD_S if 0 < speed <= 1 else max(0.5, HOLD_S / (speed or 4.0))
        if (not done and outcome is not None and not paused
                and time.perf_counter() - ended_at > hold):
            done = not start_next()

        render.poll_sinks(sinks)
        snaps = env._last
        rng = snaps["red"]["range"]
        want = max(6.0, max(rng, 800.0) * 1.6 / min(map_rect[2], map_rect[3]))
        scale += (want - scale) * 0.05

        _plan(sc, map_rect, fs, tracks, snaps, weapon, scale)
        for team in ("red", "blue"):
            start = (("blue" if team == "red" else "red") if swapped else team)
            _panel(side[team], team, names[team], notes[team], f"start.{start}",
                   snaps[team], weapon, bar_w, text_w)
            _altitude(sc, chart[team], fs, tracks[team].profile,
                      TEAM[team]["body"], env.t_max, band)

        # header: what is happening, the score, the keys
        pygame.draw.rect(sc, PANEL, (0, 0, W, head_h))
        winner = RESULT.get(outcome)
        if done:
            title, col = "done -- ESC to close", GOOD
        elif winner:
            title, col = f"{names[winner]} wins", TEAM[winner]["body"]
        elif outcome:
            title, col = outcome.upper(), WARN
        else:
            title, col = f"t {snaps['red']['t']:5.1f} s", FG
        sc.blit(f_big.render(title, True, col), (18, 10))
        info_line = (f"engagement {k}   seed {seed}   "
                     f"{'starts swapped' if swapped else 'starts as configured'}")
        sc.blit(fs.render(info_line, True, DIM), (18, 48))
        if dq:
            sc.blit(fs.render(f"   {_left(dq, names)}", True, WARN),
                    (18 + fs.size(info_line)[0], 48))
        img = f_speed.render(f"x{speed:g}" if speed else "x max", True,
                             WARN if paused else FG)
        sc.blit(img, (W - img.get_width() - 18, 8))
        if paused:
            img2 = f.render("PAUSED", True, WARN)
            sc.blit(img2, (W - img2.get_width() - 18, 8 + img.get_height()))
        pygame.draw.line(sc, (44, 50, 62), (0, head_h), (W, head_h))
        pygame.draw.line(sc, (44, 50, 62), (half, head_h), (half, head_h + panel_h))
        pygame.draw.line(sc, (44, 50, 62), (0, head_h + panel_h), (W, head_h + panel_h))

        box = _scoreboard(sc, map_rect, (f_team, f_score, fs), names, score)
        # The pale stretch waits while the hits keep coming, then catches up --
        # so a whole burst shows as one chunk, as in a fighting game.
        health = {team: snaps[team]["own_health"] for team in ("red", "blue")}
        for team in ("red", "blue"):
            if health[team] < last_hp[team]:
                wait[team] = 15
            elif wait[team] > 0:
                wait[team] -= 1
            else:
                ghost[team] = max(health[team], ghost[team] - 0.02)
            last_hp[team] = health[team]
        _health_bars(sc, map_rect, box, f, health, ghost)

        # footer: the keys
        pygame.draw.rect(sc, PANEL, (0, H - foot_h, W, foot_h))
        sc.blit(fs.render("SPACE pause   BACKSPACE next   TAB + - speed   ESC quit",
                          True, DIM), (18, H - foot_h + 5))

        render.present(sc)
        clock.tick(20.0 * speed if speed else 0)

    pygame.quit()
