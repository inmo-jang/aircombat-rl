"""Shared pygame drawing: top-down, altitude profile, cockpit.

Both the keyboard tool and the policy viewer draw through this, and that is the
point rather than a tidiness preference.  There used to be two renderers, and
the second one read the firing cone from `engagement.WEZ_ATA_DEG` -- the module
default, 15 deg -- while the task it was drawing ran at 30.  The picture was
simply wrong, in the one number the whole game turns on, and nothing caught it
because both halves were internally consistent.

So: **nothing in this package imports a weapon module.**  Everything the sight
depends on arrives in a `Weapon`, read off the environment -- the same place the
referee reads it.  A different weapon supplies a different `Weapon` and reuses
every view here.

    from aircombat_gym.tools import render
    w = render.Weapon.from_env(env)        # or Weapon(cone_deg=..., ...)
    render.topdown(surface, rect, font, own, foe, w, scale=12.0, ...)

Coordinates everywhere are the simulator's: x east, y north, h up, `psi` a
compass heading measured clockwise from north, so a screen position is
`(+sin psi, -cos psi)` and the minus on y is because pygame's y grows downward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame

from ..core.envelope import ALT_MAX_FT, ALT_MIN_FT, V_MAX_KT, V_MIN_KT

# --- palette.  Imported by the tools so it cannot drift. ---------------------
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
FOE_TRAIL = (110, 60, 60)
# The ADI's sky and ground.  Bright, because nothing is drawn on top of them.
SKY = (58, 104, 158)
GROUND = (122, 88, 52)
# The same two through a HUD.  Much darker: the symbology is drawn *over* these,
# and at ADI brightness the green ladder and the red target stop reading.
HUD_SKY = (32, 52, 76)
HUD_GROUND = (62, 48, 32)

EARTH_R_M = 6_371_000.0

FT = 0.3048
PROFILE_WINDOW_S = 120.0
FOV_HALF_DEG = 60.0                  # cockpit half-angle


@dataclass(frozen=True)
class Weapon:
    """What the sight needs.  Never defaulted from a module constant."""

    cone_deg: float
    r_min: float
    r_max: float
    muzzle_ms: float
    lock_s: float = 1.0

    @classmethod
    def from_env(cls, env) -> "Weapon":
        """Read the numbers off the environment, which is where the referee
        reads them.  Nothing in `tools/` imports a weapon module: that is what
        stopped the viewer drawing a 15 deg cone over a task running at 30."""
        e = getattr(env, "unwrapped", env)
        return cls(cone_deg=e.wez_cone_deg, r_min=e.wez_r_min,
                   r_max=e.wez_r_max, muzzle_ms=e.muzzle_ms,
                   lock_s=e.track_lock)


@dataclass
class Track:
    """One aircraft as the renderer sees it."""

    state: object
    trail: list = field(default_factory=list)        # [(x, y)]
    profile: list = field(default_factory=list)      # [(t, alt_ft)]
    dead: bool = False


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def world_vel(st) -> tuple[float, float, float]:
    """Velocity in world axes (east, north, up)."""
    vh = st.v * math.cos(st.gamma)
    return math.sin(st.track) * vh, math.cos(st.track) * vh, st.h_dot


def aim_point(me, foe, muzzle_ms: float):
    """Where the rounds have to go, and the range.  Same lead as `look()`."""
    dx, dy, dz = foe.x - me.x, foe.y - me.y, foe.h - me.h
    r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9
    mv, fv = world_vel(me), world_vel(foe)
    tof = r / muzzle_ms
    return (dx + (fv[0] - mv[0]) * tof,
            dy + (fv[1] - mv[1]) * tof,
            dz + (fv[2] - mv[2]) * tof), r


def body_axes(st):
    """(forward, right, up) unit vectors in world axes, roll included.

    Level and heading north gives forward (0,1,0), right (1,0,0), up (0,0,1);
    90 deg of right bank puts `right` straight down.
    """
    cp, sp = math.cos(st.psi), math.sin(st.psi)
    ct, stt = math.cos(st.theta), math.sin(st.theta)
    f = (sp * ct, cp * ct, stt)
    r0 = (cp, -sp, 0.0)
    u0 = (r0[1] * f[2] - r0[2] * f[1],
          r0[2] * f[0] - r0[0] * f[2],
          r0[0] * f[1] - r0[1] * f[0])
    cf, sf = math.cos(st.phi), math.sin(st.phi)
    return (f,
            tuple(r0[i] * cf - u0[i] * sf for i in range(3)),
            tuple(r0[i] * sf + u0[i] * cf for i in range(3)))


# --- symbols -----------------------------------------------------------------

def plane(sc, x, y, psi, col, size=20):
    pts = []
    for ang, rad in ((0.0, size), (2.5, size * 0.55),
                     (math.pi, size * 0.25), (-2.5, size * 0.55)):
        a = psi + ang
        pts.append((x + math.sin(a) * rad, y - math.cos(a) * rad))
    pygame.draw.polygon(sc, col, pts)


def ray(sc, cx, cy, ang, length, col, width=1):
    pygame.draw.line(sc, col, (cx, cy),
                     (cx + math.sin(ang) * length,
                      cy - math.cos(ang) * length), width)


def wez_wedge(sc, cx, cy, psi, weapon: Weapon, scale, in_wez, locked):
    """Horizontal slice of the firing cone.

    Only a slice: the referee tests the 3D angle, so a target above or below can
    sit inside this wedge and still not count.  That is what the cockpit view is
    for, and why the lock bar is the thing that settles it.
    """
    col = GOOD if (in_wez and locked) else ((150, 200, 150) if in_wez
                                            else (58, 78, 62))
    r_in, r_out = weapon.r_min / scale, weapon.r_max / scale
    half = math.radians(weapon.cone_deg)
    for side in (-1, 1):
        a = psi + side * half
        pygame.draw.line(sc, col,
                         (cx + math.sin(a) * r_in, cy - math.cos(a) * r_in),
                         (cx + math.sin(a) * r_out, cy - math.cos(a) * r_out), 1)
    for rr in (r_in, r_out):
        if rr > 3:
            rect = pygame.Rect(cx - rr, cy - rr, 2 * rr, 2 * rr)
            pygame.draw.arc(sc, col, rect, math.pi / 2 - psi - half,
                            math.pi / 2 - psi + half, 1)


def speed_vector(sc, font, cx, cy, s, v_cmd_kt):
    """Current and commanded speed along the direction of travel."""
    def px(v_kt):
        frac = (v_kt - V_MIN_KT) / (V_MAX_KT - V_MIN_KT)
        return 34 + max(0.0, min(1.0, frac)) * 210

    ux, uy = math.sin(s.track), -math.cos(s.track)
    nx, ny = uy, -ux
    l_now, l_cmd = px(s.v_kt), px(v_cmd_kt)
    pygame.draw.line(sc, (44, 50, 62), (cx + ux * 34, cy + uy * 34),
                     (cx + ux * 244, cy + uy * 244), 3)
    col = WARN if v_cmd_kt > s.v_kt + 3 else (
        ACCENT if v_cmd_kt < s.v_kt - 3 else GOOD)
    tip = (cx + ux * l_now, cy + uy * l_now)
    pygame.draw.line(sc, col, (cx + ux * 34, cy + uy * 34), tip, 7)
    cxx, cyy = cx + ux * l_cmd, cy + uy * l_cmd
    pygame.draw.line(sc, FG, (cxx + nx * 12, cyy + ny * 12),
                     (cxx - nx * 12, cyy - ny * 12), 3)
    sc.blit(font.render(f"{v_cmd_kt:.0f}", True, FG),
            (cxx + nx * 16 - 8, cyy + ny * 16 - 7))
    sc.blit(font.render(f"{s.v_kt:.0f} kt", True, col),
            (tip[0] - nx * 20 - 14, tip[1] - ny * 20 - 7))


def heading_target(sc, cx, cy, psi_cmd, radius=150):
    bx = cx + math.sin(psi_cmd) * radius
    by = cy - math.cos(psi_cmd) * radius
    pygame.draw.line(sc, (70, 60, 30), (cx, cy), (bx, by), 1)
    pygame.draw.polygon(sc, WARN, [
        (bx + math.sin(psi_cmd) * 11, by - math.cos(psi_cmd) * 11),
        (bx + math.sin(psi_cmd + 2.6) * 8, by - math.cos(psi_cmd + 2.6) * 8),
        (bx + math.sin(psi_cmd - 2.6) * 8, by - math.cos(psi_cmd - 2.6) * 8)], 2)


# --- the three views ---------------------------------------------------------

def topdown(sc, rect, font, own: Track, foe: Track | None, weapon: Weapon, *,
            scale: float, in_wez: bool = False, lock_frac: float = 0.0,
            guidance: dict | None = None, grid: bool = True):
    """Plan view, own aircraft centred.

    `guidance` is `{"psi_cmd": ..., "v_cmd_kt": ...}` when an autopilot is
    flying and None when a stick is; the difference is what gets overlaid.
    """
    x0, y0, w, h = rect
    sc.fill(BG, rect)
    old = sc.get_clip()
    sc.set_clip(pygame.Rect(rect))
    s = own.state
    cx, cy = x0 + w // 2, y0 + h // 2

    def px(x, y):
        return int(cx + (x - s.x) / scale), int(cy - (y - s.y) / scale)

    if grid:
        step = 1000.0 / scale
        if step >= 12:
            gx = x0 + (-s.x / scale) % step
            while gx < x0 + w:
                pygame.draw.line(sc, GRID, (int(gx), y0), (int(gx), y0 + h))
                gx += step
            gy = y0 + (s.y / scale) % step
            while gy < y0 + h:
                pygame.draw.line(sc, GRID, (x0, int(gy)), (x0 + w, int(gy)))
                gy += step

    for tr, col in ((foe.trail if foe else [], FOE_TRAIL), (own.trail, TRAIL)):
        if len(tr) > 1:
            stride = max(1, len(tr) // 400)
            pts = [px(x, y) for x, y in tr[::stride]]
            if len(pts) > 1:
                pygame.draw.lines(sc, col, False, pts, 2)

    if guidance is not None:
        speed_vector(sc, font, cx, cy, s, guidance["v_cmd_kt"])
        heading_target(sc, cx, cy, guidance["psi_cmd"])
    else:
        # nose against track: bank rotates angle of attack into the horizontal
        # plane and the two split by as much as 27 deg in hard manoeuvring
        ray(sc, cx, cy, s.psi, 150, (90, 130, 90))
        ray(sc, cx, cy, s.track, 110, ACCENT)

    if foe is not None:
        locked = lock_frac >= 1.0
        wez_wedge(sc, cx, cy, s.psi, weapon, scale, in_wez, locked)
        fs = foe.state
        fx, fy = px(fs.x, fs.y)
        col = (120, 120, 120) if foe.dead else (WARN if lock_frac > 0 else BAD)
        plane(sc, fx, fy, fs.psi, col, 16)
        if locked and not foe.dead:
            pygame.draw.circle(sc, WARN, (fx, fy), 22, 2)
        (ax, ay, _), rng = aim_point(s, fs, weapon.muzzle_ms)
        sc.blit(font.render(f"{rng:,.0f} m", True, col), (fx + 18, fy - 6))
        # the pipper is where to point, and it is not where he is
        ppx, ppy = int(cx + ax / scale), int(cy - ay / scale)
        pcol = WARN if in_wez else (150, 170, 200)
        pygame.draw.line(sc, (90, 100, 120), (fx, fy), (ppx, ppy), 1)
        pygame.draw.circle(sc, pcol, (ppx, ppy), 9, 2)
        pygame.draw.circle(sc, pcol, (ppx, ppy), 2)

    plane(sc, cx, cy, s.psi, ACCENT, 20)
    sc.blit(font.render(f"{1000.0 / scale:.0f} px = 1 km", True, DIM),
            (x0 + 12, y0 + h - 20))
    sc.set_clip(old)


def profile(sc, rect, font, own: Track, foe: Track | None = None, *,
            alt_cmd_ft: float | None = None, window_s: float = PROFILE_WINDOW_S,
            autoscale: bool = False):
    """Altitude against *time*.  The axis a plan view cannot show.

    Time rather than ground distance: in a hard turn the ground track doubles
    back on itself and the trace becomes unreadable, while what you want is how
    fast height is being spent and bought back.

    `autoscale=False` keeps the envelope in frame (ground, floor, ceiling),
    which is what hand-flying wants.  `autoscale=True` fits the traces, which is
    what a task with both aircraft pinned to one altitude needs -- otherwise
    both lines sit on top of each other in the middle of an empty chart.
    """
    x0, y0, w, h = rect
    pygame.draw.rect(sc, (20, 23, 30), rect)
    series = list(own.profile)
    if not series:
        return
    fseries = list(foe.profile) if foe else []

    if autoscale:
        alts = [a for _, a in series] + [a for _, a in fseries]
        lo, hi = min(alts), max(alts)
        pad = max(300.0, (hi - lo) * 0.35)
        lo, hi = lo - pad, hi + pad
    else:
        peak = max(a for _, a in series)
        lo, hi = 0.0, max(35000.0, math.ceil((peak + 2000.0) / 5000.0) * 5000.0)

    def ypx(ft):
        return y0 + h - (max(lo, min(hi, ft)) - lo) / max(hi - lo, 1e-6) * h

    if autoscale:
        step = 1000.0
        first = int(lo // step) * step
        for ft in [first + i * step for i in range(int((hi - lo) / step) + 2)]:
            if lo <= ft <= hi:
                yy = int(ypx(ft))
                pygame.draw.line(sc, GRID, (x0, yy), (x0 + w, yy), 1)
                sc.blit(font.render(f"{ft/1000:.0f}k", True, DIM),
                        (x0 + 4, yy - 14))
    else:
        for ft, col, lbl in ((ALT_MAX_FT, (60, 70, 90), "30k ceiling"),
                             (ALT_MIN_FT, (60, 70, 90), "5k floor"),
                             (0.0, (110, 50, 50), "ground")):
            yy = int(ypx(ft))
            pygame.draw.line(sc, col, (x0, yy), (x0 + w, yy), 1)
            sc.blit(font.render(lbl, True, col), (x0 + 4, yy - 14))
        sc.blit(font.render(f"{hi / 1000:.0f}k", True, DIM), (x0 + 4, y0 + 2))

    if alt_cmd_ft is not None:
        yc = int(ypx(alt_cmd_ft))
        pygame.draw.line(sc, WARN, (x0, yc), (x0 + w, yc), 1)
        sc.blit(font.render(f"target {alt_cmd_ft:,.0f}", True, WARN),
                (x0 + 4, yc + 2))

    t1 = series[-1][0]
    t0 = min(series[0][0], t1 - window_s)
    span = max(t1 - t0, 1e-3)
    for data, col, dot in ((fseries, FOE_TRAIL, False), (series, ACCENT, True)):
        if len(data) < 2:
            continue
        stride = max(1, len(data) // 600)
        pts = [(x0 + (t - t0) / span * w, ypx(ft)) for t, ft in data[::stride]]
        if len(pts) > 1:
            pygame.draw.lines(sc, col, False, pts, 2)
        if dot:
            pygame.draw.circle(sc, FG, (int(pts[-1][0]), int(pts[-1][1])), 3)
    sc.blit(font.render(f"altitude, last {window_s:.0f} s"
                        f"   now {own.state.h_ft:,.0f} ft", True, DIM),
            (x0 + w - 250, y0 + 4))


def cockpit(sc, rect, fonts, own: Track, foe: Track | None, weapon: Weapon, *,
            in_wez: bool = False, lock_frac: float = 0.0,
            fov_half_deg: float = FOV_HALF_DEG):
    """Forward view from the nose -- the view that shows the 3D angle.

    A plan view flattens elevation away, so a target 1,000 ft high at 800 m
    looks dead ahead from above and is 21 deg off the gun line.  Here the firing
    cone is a fixed circle on the canopy and the pipper is either inside it or
    it is not.
    """
    f, fs_font = fonts
    x0, y0, w, h = rect
    cx, cy = x0 + w // 2, y0 + h // 2
    old = sc.get_clip()
    sc.set_clip(pygame.Rect(rect))
    s = own.state
    px_per_rad = (w / 2) / math.radians(fov_half_deg)

    # --- sky and ground -----------------------------------------------------
    # What the canopy actually frames, and it does the job a pitch number
    # cannot: inverted is obvious at a glance.  Same construction as the ADI --
    # the world rotates and the aircraft stays put -- on a rectangle instead of
    # a disc.
    #
    # The horizon is not at zero pitch.  From 20,000 ft it sits about 2.5 deg
    # *below* the local horizontal (`acos(R/(R+h))`), which is 9 px here.  Small,
    # but it is the difference between a horizon that tracks altitude and one
    # that is painted on.
    ux, uy = math.cos(s.phi), -math.sin(s.phi)         # along the horizon
    vx, vy = math.sin(s.phi), math.cos(s.phi)          # perpendicular, downward
    depression = math.sqrt(max(0.0, 2.0 * max(s.h, 0.0) / EARTH_R_M))
    dy_h = (s.theta + depression) * px_per_rad
    hx, hy = cx + vx * dy_h, cy + vy * dy_h

    sc.fill(HUD_SKY, rect)
    L = 3 * max(w, h)
    pygame.draw.polygon(sc, HUD_GROUND, [
        (hx - ux * L, hy - uy * L), (hx + ux * L, hy + uy * L),
        (hx + ux * L + vx * L, hy + uy * L + vy * L),
        (hx - ux * L + vx * L, hy - uy * L + vy * L)])
    pygame.draw.line(sc, (150, 170, 190), (hx - ux * L, hy - uy * L),
                     (hx + ux * L, hy + uy * L), 2)

    # --- pitch ladder -------------------------------------------------------
    for deg in (-20, -10, 0, 10, 20):
        dy = (s.theta - math.radians(deg)) * px_per_rad
        lx = 90 if deg else 150
        ca, sa = math.cos(-s.phi), math.sin(-s.phi)
        for sgn in (-1, 1):
            p1 = (cx + sgn * lx * 0.25 * ca - dy * sa,
                  cy + sgn * lx * 0.25 * sa + dy * ca)
            p2 = (cx + sgn * lx * ca - dy * sa, cy + sgn * lx * sa + dy * ca)
            pygame.draw.line(sc, (150, 200, 165) if deg == 0 else (120, 165, 135),
                             p1, p2, 2 if deg == 0 else 1)
        if deg:
            sc.blit(fs_font.render(f"{deg:+d}", True, (120, 165, 135)),
                    (cx + lx * ca - dy * sa - 6, cy + lx * sa + dy * ca - 6))

    locked = lock_frac >= 1.0
    ccol = GOOD if (in_wez and locked) else ((150, 200, 150) if in_wez
                                             else (60, 80, 66))
    cone_px = math.radians(weapon.cone_deg) * px_per_rad
    pygame.draw.circle(sc, ccol, (cx, cy), int(cone_px), 1)
    pygame.draw.line(sc, ccol, (cx - 12, cy), (cx + 12, cy), 1)
    pygame.draw.line(sc, ccol, (cx, cy - 12), (cx, cy + 12), 1)

    if foe is not None:
        fwd, right, up = body_axes(s)
        fst = foe.state
        (ax, ay, az), rng = aim_point(s, fst, weapon.muzzle_ms)
        d = (fst.x - s.x, fst.y - s.y, fst.h - s.h)

        def project(v):
            a = v[0] * fwd[0] + v[1] * fwd[1] + v[2] * fwd[2]
            if a <= 1.0:
                return None
            r_ = v[0] * right[0] + v[1] * right[1] + v[2] * right[2]
            u_ = v[0] * up[0] + v[1] * up[1] + v[2] * up[2]
            return (cx + math.atan2(r_, a) * px_per_rad,
                    cy - math.atan2(u_, a) * px_per_rad)

        tgt, pip = project(d), project((ax, ay, az))
        if tgt and pip:
            pygame.draw.line(sc, (90, 100, 120), tgt, pip, 1)
        if tgt:
            col = (120, 120, 120) if foe.dead else BAD
            pygame.draw.circle(sc, col, (int(tgt[0]), int(tgt[1])), 7, 2)
            sc.blit(fs_font.render(f"{rng:,.0f} m", True, col),
                    (tgt[0] + 11, tgt[1] - 7))
        if pip:
            pygame.draw.circle(sc, WARN if in_wez else (150, 170, 200),
                               (int(pip[0]), int(pip[1])), 5, 2)
        if tgt is None:
            sc.blit(f.render("target is behind", True, DIM), (cx - 62, cy + 44))

    sc.set_clip(old)
    pygame.draw.rect(sc, (44, 50, 62), rect, 1)
    sc.blit(fs_font.render(f"COCKPIT   FOV +-{fov_half_deg:.0f}   "
                           f"cone +-{weapon.cone_deg:.0f} deg", True, DIM),
            (x0 + 10, y0 + 8))


# =============================================================================
# layout and readout
# =============================================================================

W, H = 1300, 860
MAP_W = 880                 # left column: plan view over the altitude strip
TOP_H = 566
PROF_TOP = 574
COCKPIT_H = 300             # right column: cockpit over the readout


class Layout:
    """Where every view goes.  One definition, so the tools cannot drift apart.

    `manual_operation` and `auto_operation` differ in what they *put* in the
    readout, never in where anything sits.
    """

    W, H = W, H
    topdown = (0, 0, MAP_W, TOP_H)
    profile = (12, PROF_TOP + 6, MAP_W - 24, H - PROF_TOP - 18)
    cockpit = (MAP_W, 0, W - MAP_W, COCKPIT_H)
    readout = (MAP_W, COCKPIT_H, W - MAP_W, H - COCKPIT_H)

    @staticmethod
    def dividers(sc):
        pygame.draw.line(sc, (44, 50, 62), (MAP_W, 0), (MAP_W, H))
        pygame.draw.line(sc, (44, 50, 62), (0, PROF_TOP - 4), (MAP_W, PROF_TOP - 4))


def fonts():
    return (pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True),
            pygame.font.SysFont("consolas,menlo,monospace", 16),
            pygame.font.SysFont("consolas,menlo,monospace", 13))


# --- readout items.  Content is data; the Readout decides where it lands. ----

def text(s, col=FG, small=False, big=False):
    return ("text", s, col, small, big)


def gap(px=8):
    return ("gap", px, None, False, False)


def head(s):
    return ("head", s, DIM, True, False)


def bar(label, frac, col, value="", width=200):
    return ("bar", (label, frac, col, value, width), None, False, False)


def marked_bar(label, frac, col, mark, value="", width=200):
    """A bar with a tick on it -- 'what you have' against 'what is available'."""
    return ("mbar", (label, frac, col, mark, value, width), None, False, False)


def custom(fn, height):
    """`fn(surface, x, y, width)` -- for the stick box and anything else odd."""
    return ("custom", (fn, height), None, False, False)


class Readout:
    """The right-hand panel.  Give it items; it fits them in the space it has.

    The tools used to lay this out by hand, adding up pixel heights in their
    heads.  That failed three times in one afternoon -- most visibly when the
    cockpit view took 300 px and the flight-test instruments carried on drawing
    over the key help.  So the panel measures its own content and tightens the
    line spacing until it fits, down to a floor; past that it says so on screen
    rather than quietly overwriting itself.
    """

    PAD_X = 18
    MIN_SCALE = 0.62

    def __init__(self, screen, fonts_, rect=None):
        self.sc = screen
        self.f_big, self.f, self.fs = fonts_
        self.rect = rect or Layout.readout

    def _h(self, item, scale):
        kind, payload = item[0], item[1]
        if kind == "gap":
            return max(2, int(payload * scale))
        if kind == "custom":
            return payload[1]
        if kind == "head":
            return int(17 * scale)
        if kind in ("bar", "mbar"):
            return int(34 * scale)
        return int((26 if item[4] else (15 if item[3] else 19)) * scale)

    def draw(self, items, pinned=()):
        """`pinned` is drawn flush to the bottom -- key help, clock, that sort."""
        x0, y0, w, h = self.rect
        sc = self.sc
        pygame.draw.rect(sc, PANEL, self.rect)
        x = x0 + self.PAD_X

        pin_h = sum(self._h(i, 1.0) for i in pinned)
        avail = h - pin_h - 12
        need = sum(self._h(i, 1.0) for i in items)
        scale = 1.0 if need <= avail else max(self.MIN_SCALE, avail / max(need, 1))

        y = y0 + 12
        for item in items:
            self._one(item, x, y, w, scale)
            y += self._h(item, scale)
        if need > avail and scale <= self.MIN_SCALE + 1e-6:
            sc.blit(self.fs.render("... panel is out of room", True, BAD),
                    (x, y0 + h - pin_h - 16))

        y = y0 + h - pin_h - 4
        for item in pinned:
            self._one(item, x, y, w, 1.0)
            y += self._h(item, 1.0)

    def _one(self, item, x, y, w, scale):
        kind, payload, col, small, big = item
        sc = self.sc
        if kind == "gap":
            return
        if kind == "custom":
            payload[0](sc, x, y, w - 2 * self.PAD_X)
            return
        if kind in ("text", "head"):
            font = self.f_big if big else (self.fs if small else self.f)
            sc.blit(font.render(payload, True, col), (x, y))
            return
        if kind == "bar":
            label, frac, bcol, value, bw = payload
            sc.blit(self.fs.render(label, True, DIM), (x, y))
            self._rect_bar(x, y + int(15 * scale), bw, frac, bcol)
            if value:
                sc.blit(self.fs.render(value, True, bcol),
                        (x + bw + 10, y + int(14 * scale)))
            return
        label, frac, bcol, mark, value, bw = payload
        sc.blit(self.fs.render(label, True, DIM), (x, y))
        by = y + int(15 * scale)
        self._rect_bar(x, by, bw, frac, bcol)
        mx = x + int(bw * max(0.0, min(1.0, mark)))
        pygame.draw.line(sc, FG, (mx, by - 2), (mx, by + 14), 2)
        if value:
            sc.blit(self.fs.render(value, True, bcol), (x + bw + 10, by - 1))

    def _rect_bar(self, x, y, w, frac, col, h=13):
        pygame.draw.rect(self.sc, (45, 50, 62), (x, y, w, h), border_radius=2)
        f = max(0.0, min(1.0, frac))
        if f > 0:
            pygame.draw.rect(self.sc, col, (x, y, int(w * f), h), border_radius=2)


def stick_box(sc, font, x, y, size, ctl):
    """Aileron/elevator as a square, rudder as a bar under it."""
    pygame.draw.rect(sc, (20, 23, 30), (x, y, size, size))
    pygame.draw.line(sc, GRID, (x + size // 2, y), (x + size // 2, y + size))
    pygame.draw.line(sc, GRID, (x, y + size // 2), (x + size, y + size // 2))
    px = x + size / 2 + ctl["aileron"] * size / 2
    py = y + size / 2 + ctl["elevator"] * size / 2
    pygame.draw.circle(sc, WARN, (int(px), int(py)), 6)
    ry = y + size + 6
    pygame.draw.rect(sc, (45, 50, 62), (x, ry, size, 8))
    rx = x + size / 2 + ctl["rudder"] * size / 2
    pygame.draw.rect(sc, ACCENT, (int(rx) - 3, ry, 6, 8))
    sc.blit(font.render("stick / rudder", True, DIM), (x, ry + 12))


# =============================================================================
# what both tools do around the drawing
# =============================================================================

def open_window(title: str):
    """One window, one size, one set of fonts.  Returns (screen, fonts, clock).

    The two tools opened their own and picked their own font names -- close
    enough to look the same and different enough that a label fitted in one and
    clipped in the other.
    """
    pygame.init()
    screen = pygame.display.set_mode((Layout.W, Layout.H))
    pygame.display.set_caption(f"aircombat - {title}")
    return screen, fonts(), pygame.time.Clock()


def push(track: Track, t: float, cap: int = 2400) -> None:
    """Record one frame of a track's history and forget the oldest.

    `cap` is a frame count; 2,400 is a full 120 s episode at 20 Hz.
    """
    st = track.state
    track.trail.append((st.x, st.y))
    track.profile.append((t, st.h_ft))
    for buf in (track.trail, track.profile):
        if cap and len(buf) > cap:
            del buf[0]


def world_views(sc, own: Track, foe: Track | None, weapon: Weapon, *,
                scale: float, in_wez: bool = False, lock_frac: float = 0.0,
                guidance: dict | None = None, alt_cmd_ft: float | None = None,
                profile_autoscale: bool = False, fs_font=None):
    """The three pictures, in their fixed places.  The readout is the caller's.

    Everything above the readout is identical between hand-flying and watching
    a policy, so it is one call.  What differs -- a stick panel against a
    scoreboard -- is what each tool passes to `Readout`.
    """
    sc.fill(BG)
    topdown(sc, Layout.topdown, fs_font, own, foe, weapon, scale=scale,
            in_wez=in_wez, lock_frac=lock_frac, guidance=guidance)
    profile(sc, Layout.profile, fs_font, own, foe if profile_autoscale else None,
            alt_cmd_ft=alt_cmd_ft, autoscale=profile_autoscale)
    Layout.dividers(sc)


def fit_scale(current: float, range_m: float, *, ease: float = 0.05,
              floor: float = 6.0, margin: float = 2.4) -> float:
    """Keep both aircraft in the plan view without the zoom juddering."""
    want = max(floor, max(range_m, 800.0) * margin
               / min(Layout.topdown[2], Layout.topdown[3]))
    return current + (want - current) * ease


# --- TacView sinks -----------------------------------------------------------

def poll_sinks(sinks) -> None:
    """Let a viewer attach even while paused, between episodes or after a crash.

    A sink only accepts inside `frame()`, so a tool that stops stepping stops
    accepting -- and TacView, retrying, never gets in.
    """
    for k in sinks:
        if hasattr(k, "poll"):
            k.poll()


def emit_frame(sinks, t: float, own_state, foe_state=None, *,
               own_id="101", foe_id="102", locked: bool | None = None) -> None:
    """Send one ACMI frame to every sink."""
    if not sinks:
        return
    from ..core.tacview import state_to_object
    objs = [state_to_object(own_state, own_id, "F-16C", "Blue")]
    if foe_state is not None:
        objs.append(state_to_object(foe_state, foe_id, "F-16C", "Red"))
        if locked is not None:
            objs[0]["locked"] = foe_id if locked else None
    for k in sinks:
        k.frame(t, objs)


def emit_event(sinks, note: str, *ids) -> None:
    for k in sinks:
        k.event(note, *(ids or ("101", "102")))
