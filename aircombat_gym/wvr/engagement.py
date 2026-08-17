"""Gun engagement geometry and the weapon employment zone.

The three canonical WVR quantities:

  ATA  antenna train angle -- angle between my nose and the line of sight.
       "How well am I pointing at him."  Zero means he is straight ahead.
  AA   aspect angle -- angle between his tail and the line of sight.
       "How close am I to his six."  Zero means I am directly behind him.
  R    range.

Perfect attack is ATA = AA = 0.  Perfect defence is ATA = AA = 180.
The geometry is honest 3D -- altitude is a real axis now, so dz is in every term.

WHY THESE NUMBERS.  The aircraft carries an M61A1: 20 mm, 6,000 rounds/min,
511 rounds, muzzle velocity about 1,050 m/s.  That is 5.1 seconds of trigger
time in the whole jet, and a burst that connects for a few tenths of a second
is lethal.  So the model is: a *held* tracking solution kills in about a second,
and everything else falls off hard.

  cone     `WEZ_ATA_DEG`, +-15 deg.  A gun's dispersion is a few milliradians;
           this angle is not dispersion, it is how far off the tracking solution
           can be and still put rounds through the target.  Damage falls
           linearly to zero at the edge, so meaningful hits happen within a few
           degrees and the rest is grazing.  **Every task overrides it to 30**,
           for exploration rather than realism -- see `envs/circular.py`.
           10 deg was tried and is too tight to fly by hand (see below).
  range    150 m to 1,500 m.  Not "how far a bullet reaches" -- this is the
           denominator of the falloff, and 1,500 m is what makes a 600 m shot
           worth 0.67 instead of 0.47.  Six hundred metres is a good gun range
           and should not be penalised like a long one; meanwhile the linear
           falloff still takes a genuine 1,500 m shot to zero.  150 m is a real
           minimum: closer than that and you are eating your own debris.
  lead     rounds take R/1,050 s to arrive, so the aim point is where the target
           *will be*, not where it is.  At 600 m that is half a second, and a
           450 kt target moves 116 m in it -- 11 deg of lead.  Without this term
           pure pursuit is a perfect gun solution, which is wrong and is the
           likeliest reason the `lead` bot never beat `pursuit` when the game
           was 2D.
  aspect   a crossing target is much harder than one flying away from you: the
           line-of-sight rate is high, the lead angle is large and the tracking
           window is short.  ASPECT_FLOOR is what a pure beam shot is worth.

MEASURED, because the first attempt at "realistic" was unplayable.  A lead
pursuit autopilot chasing the circler, best speed of 400-475 kt, time to kill:

    cone   R_MAX   from 1.5 km   2.5 km   4.0 km
    10 deg  1000 m    no kill     77.3 s   126.3 s
    15      1000      40.4        74.6     124.1
    15      1500      37.9        46.1      83.5
    20      1500      36.4        45.4      80.8
    30      1500      35.1        44.5      78.9

Range dominates and the cone barely matters past 20 deg, which is why the
falloff denominator is 1,500 m: it keeps the lead requirement biting -- pure
pursuit still does much worse than lead -- without making the range term punish
good shots.  The track lock is not a constraint at all here (0.6 s and 0.3 s
give 36.4 s alike).

The first version of this file used 10 deg and 1,000 m on the grounds that they
were realistic, and a human flying it by hand for three minutes never once got
inside the envelope -- closest approach 647 m, so the closing was fine and the
*aiming* was impossible.  "Does it match reality" is not the same question as
"does it work as a game", and both have to be asked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

WEZ_ATA_DEG = 15.0     # tracking tolerance, not dispersion
WEZ_R_MIN = 150.0      # own debris / overshoot
WEZ_R_MAX = 1500.0     # falloff denominator, not a bullet's reach
MUZZLE_MS = 1050.0     # M61A1 20 mm, for the time-of-flight lead

# Health per second with the pipper on the aim point at minimum range and dead
# astern.  The three falloffs multiply, so a realistic shot is worth much less:
# 400 m, 2 deg off, 20 deg of aspect works out to 0.54/s -- about two seconds.
DAMAGE_RATE = 1.0

ASPECT_FLOOR = 0.15    # what a pure beam snapshot is worth


@dataclass(frozen=True)
class Engagement:
    """One aircraft's view of the fight.  Everything is egocentric."""

    ata: float            # [rad] 0..pi, to the target itself -- for observation
    ata_signed: float     # [rad] -pi..pi, positive = target is to the right
    ata_lead: float       # [rad] to the aim point -- this is what the gun needs
    aa: float             # [rad] 0..pi
    r: float              # [m]
    r_dot: float          # [m/s], positive = opening
    in_wez: bool
    damage_rate: float    # health per second I am inflicting right now
    # where to point, relative to me, in the horizontal plane.  A real fighter
    # computes this and draws it on the HUD as the pipper; asking a pilot to
    # lead a target without showing the lead is not a difficulty setting, it is
    # a missing instrument.
    aim_dx: float
    aim_dy: float


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _angle_to(dx, dy, dz, nx, ny, nz) -> float:
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    if r < 1e-6 or n < 1e-6:
        return 0.0
    return math.acos(max(-1.0, min(1.0, (dx * nx + dy * ny + dz * nz) / (r * n))))


def look(me, foe, cone_deg: float = WEZ_ATA_DEG,
         damage_rate: float = DAMAGE_RATE) -> Engagement:
    """`cone_deg` and `damage_rate` are per-task, so they are arguments.

    Arguments rather than module constants a task reassigns: under
    `SubprocVecEnv(start_method="spawn")` each worker re-imports the module and
    gets the default back, which once made two "different" runs identical.
    """
    dx, dy, dz = foe.x - me.x, foe.y - me.y, foe.h - me.h
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    if r < 1e-6:
        return Engagement(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, 0.0, 0.0)

    # The gun line is the nose, not the velocity vector: rounds leave along the
    # boresight, and in hard manoeuvring the two differ by as much as 27 deg
    # because bank rotates the angle of attack into the horizontal plane.
    nx, ny, nz = math.sin(me.psi), math.cos(me.psi), math.sin(me.theta)
    ata = _angle_to(dx, dy, dz, nx, ny, nz)

    fx, fy, fz = math.sin(foe.psi), math.cos(foe.psi), math.sin(foe.theta)
    aa = _angle_to(dx, dy, dz, fx, fy, fz)

    # signed bearing in the horizontal plane, which is what steering needs
    ata_signed = _wrap(math.atan2(dx, dy) - me.psi)

    # closure along the line of sight
    r_dot = ((dx * (foe.vx - me.vx) + dy * (foe.vy - me.vy)
              + dz * (foe.vz - me.vz)) / r)

    # --- lead: aim where he will be when the rounds arrive -------------------
    # Straight-line prediction over the time of flight.  A real lead-computing
    # sight does better against a manoeuvring target, but the error that matters
    # here is the one pure pursuit makes, and this captures it.
    tof = r / MUZZLE_MS
    ax = dx + foe.vx * tof - me.vx * tof
    ay = dy + foe.vy * tof - me.vy * tof
    az = dz + foe.vz * tof - me.vz * tof
    ata_lead = _angle_to(ax, ay, az, nx, ny, nz)

    lead_deg = math.degrees(ata_lead)
    inside = lead_deg <= cone_deg and WEZ_R_MIN <= r <= WEZ_R_MAX
    dmg = 0.0
    if inside:
        # three factors, all multiplying: how well the solution is held, how
        # close, and how square to his tail.  The last is what makes the six
        # worth reaching and the beam worth denying.
        f_aspect = ASPECT_FLOOR + (1.0 - ASPECT_FLOOR) * max(0.0, math.cos(aa))
        dmg = (damage_rate
               * (1.0 - lead_deg / cone_deg)
               * (1.0 - (r - WEZ_R_MIN) / (WEZ_R_MAX - WEZ_R_MIN))
               * f_aspect)
    return Engagement(ata, ata_signed, ata_lead, aa, r, r_dot, inside, dmg,
                      ax, ay)
