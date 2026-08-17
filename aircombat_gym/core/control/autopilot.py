"""Autopilot: (heading, speed, altitude) targets -> stick and throttle.

This is the layer a student policy talks to.  The usual way to build it is two
stages -- targets into physical commands (Nz, roll rate, throttle), then those
into control surfaces, often by LQR.  Here it is three cascaded loops instead,
because the F-16's own FLCS already provides roll damping, a g-command pitch
channel and an alpha limiter at 120 Hz, so what is needed on top is thin.  It
must stay thin: adding p/q derivative terms on top of loops that are already
damped makes the response sluggish, which is the suspected cause of the
throwaway PID's 17.3 s turns.

    heading   psi_cmd -> phi_cmd (bank ceiling)  -> aileron
    altitude  alt_cmd -> hdot_cmd -> nz_cmd      -> elevator
    speed     v_cmd   -> throttle (<= throttle_cap)

The bank ceiling is not a constant, and that is the important part.  A level
turn needs n = 1/cos(phi); ask for a bank the current speed and height cannot
support and the aircraft simply runs out of lift -- which is how the throwaway
PID lost 3,278 ft.  A waypoint follower can get away with a fixed ceiling
because nothing asks it to hold an altitude while manoeuvring; this does.

So every step the available lift is *budgeted*: the altitude correction takes
its share first and the bank ceiling is whatever is left, meaning the aircraft
gives up turn rate rather than height.  A fixed 0.9 * n_max ceiling is not
enough on its own -- it leaves 0.6 g spare at corner speed but only 0.25 g at
the bottom of the speed band, and measured 192 ft of drift with the elevator
saturated 30 % of the time.

Measured with this in place: 300 s of adversarial 20 Hz random commands hold
altitude to a median of 17 ft, p99 of 70 ft and a worst transient of 104 ft,
|gamma| under 8.2 deg, no departures.  k_h above about 0.6 goes unstable.
Deliberate climbs and dives are a different matter and use the full 250 ft/s:
40 s of command takes 20,000 ft to 29,719 or down to 10,645.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..envelope import (ALT_MAX_FT, ALT_MIN_FT, H0_FT, N_RESERVE, THROTTLE_CAP,
                        V_MAX_KT, V_MIN_KT, n_max)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class AutopilotGains:
    # --- heading -> bank ------------------------------------------------------
    #
    #   phi_cmd = clip(k_psi * psi_err - k_psi_r * r, +-max_bank)
    #
    # What a waypoint follower normally does: hold the bank ceiling until the
    # error is small, then come off it linearly.
    #
    # There used to be an alternative law here, a taper that mapped the error
    # onto *turn rate* instead of bank -- atan(((|err|/30 deg)^0.7) * tan(mb)) --
    # so that a 15 deg command turned at a defined fraction of a 30 deg one.  It
    # existed for gate H3, a 2D-era requirement that the action channels differ
    # in *sustained* rate, and it is gone with that gate.  The measurement is
    # kept because it is what chose k_psi.
    #
    # 90 deg step at 450 kt / 20,000 ft.  The ideal bang-bang floor is 7.3 s
    # (90 deg at the 12.3 deg/s ceiling), and the first 60 deg costs 6.9 s under
    # every law because that part is the airframe, not the loop -- all any of
    # this changes is the tail.
    #
    #   law            ->30 deg   ->10   ->2    overshoot   15 vs 30 rate
    #   taper            6.9 s    11.2   14.8      0.3 deg      1.52x
    #   prop k=5         6.9      12.2   20.1      0.0          1.35x
    #   prop k=8         6.9       9.4   15.8      0.0          1.00x
    #   prop k=12        6.9       9.4   14.3      0.2          1.00x
    #   prop k=20        6.9       9.4   12.3      0.5          1.00x
    #   prop k=40        6.9       9.4   10.3      6.2          1.00x
    #
    # Two things fell out of that.  A mild proportional gain is *slower* than
    # the taper, not faster -- a linear ramp off the ceiling has a longer tail
    # than a 0.7 power does, so the gain has to be stiff to win.  And
    # distinctness and settling time turn out to be the same knob pointed in
    # opposite directions: telling 15 from 30 requires the ramp to extend past
    # 30 deg of error, which is exactly the region the tail lives in.
    #
    # k=20 chosen: fastest across the envelope (12-31 % quicker than the taper
    # at 7k/20k/29k ft x 320/450/620 kt) with overshoot never above 0.7 deg and
    # no change in altitude hold.  k=40 gets another 2 s and overshoots 6.2 deg.
    #
    # What the actions lose by this is nothing that matters: a tap of +15 still
    # turns 15 deg where +30 turns 30, and a policy can duty-cycle for an
    # intermediate rate.  They coincide only while *held*, because holding
    # re-derives the target every step and pins the error at the delta, so any
    # gain stiff enough to fly well saturates at both (measured: identical,
    # 78.6 deg of bank and 8.73 deg/s).
    #
    # What is NOT optional is that the clip is `max_bank(V, h)` and
    # not a constant.  A fixed ceiling is what let the throwaway PID command
    # 70 deg at 250 kt, which needs 2.92 g against an available 1.93, and lose
    # 3,278 ft.
    #
    # A 180 deg reversal still overshoots about 10 deg at the bottom of the
    # speed band, and yaw-rate damping does not fix it (measured flat at 9.5-9.7
    # deg for k_psi_r from 0.1 to 1.5).  The cause is roll-out time: at 320 kt it
    # takes ~2 s to come off 67 deg of bank and the aircraft sweeps another 14 deg
    # doing it, so holding the ceiling until 3.5 deg of error guarantees going
    # past.  Killing it needs the law to anticipate the roll-out, not more
    # damping.  Left alone because the scoring interface never issues a step
    # like this -- actions are +-15/30 deg re-derived every 50 ms, so the target
    # is never approached from far away and there is nothing to overshoot.
    # Only hand-flying in TARGET mode can set up the condition.
    #
    # k_psi_r 0.6 is still worth having: same overshoot, but it damps the
    # recovery afterwards, and settling at 320 kt goes 31.6 -> 25.4 s.  Above
    # 1.0 it starts costing time at 450 kt (21.9 -> 23.5 s).
    k_psi: float = 20.0            # rad/rad; holds the ceiling to ~4 deg of error
    k_psi_r: float = 0.6           # yaw-rate damping
    # bank -> aileron (FLCS takes a roll-rate command, so damping stays light)
    k_phi: float = 3.0
    k_p: float = 0.35
    # altitude -> climb rate -> Nz
    k_h: float = 0.50          # 1/s, on feet of error
    k_hd: float = 0.045        # g per ft/s
    hdot_limit: float = 250.0  # ft/s -- 80 was sized for the locked plane, where
                               # the loop only ever corrected drift.  Climbing
                               # and diving on purpose needs real authority.
    # Nz -> elevator
    k_nz: float = 0.30
    k_nzi: float = 0.60
    i_nz_limit: float = 2.5
    # --- speed -> throttle ---------------------------------------------------
    # Both terms earn their place, and the gain was set by measurement.
    #
    # k_v was 0.010, and that was too soft to saturate at the errors that
    # matter.  Big steps saturated anyway, but a 50 kt error only asked for
    # 0.45 + 0.5 = 0.95 throttle -- and near the top of the speed band excess
    # thrust is small, so those last few percent are most of the acceleration.
    # Measured, time to come within 5 kt:
    #
    #                600->650 kt   450->650   300->650   hold 450 kt
    #   k_v 0.010    not in 90 s      31.6 s     46.0 s    -0.02 kt
    #   k_v 0.05           10.9 s     25.6 s     39.2 s    +0.02
    #   k_v 0.10            8.6 s     24.2 s     37.7 s    +0.07
    #   k_v 0.25            8.6 s     24.2 s     37.8 s    +0.07
    #   pure P, k 0.25      8.8 s     24.4 s     38.0 s    -1.64
    #
    # 0.10 saturates at about 5 kt of error, which is where the useful range
    # ends -- 0.25 buys nothing because it is already hard against the stop.
    # Throttle chatter is unchanged (std 0.0013 holding 450 kt, same as before).
    #
    # The bias and the integral are both needed.  Drop the bias and the integral
    # has to supply cruise throttle on its own, hits its limit and leaves
    # 1.75 kt of steady error; drop the integral and there is 1.64 kt of steady
    # error wherever bias happens not to match drag.
    thr_bias: float = 0.45
    k_v: float = 0.10          # per knot of error
    k_vi: float = 0.004        # per knot-second
    i_v_limit: float = 60.0
    # bank ceiling used by the Nz feedforward; 1/cos blows up past this
    phi_ff_limit_deg: float = 84.0
    # Load-factor budget.  There is only so much g available, and both the turn
    # (n = 1/cos phi) and the altitude correction have to come out of it.
    # Altitude is served first and the bank ceiling takes what is left, so the
    # aircraft gives up turn rate rather than the plane it is locked to.
    # Without this the altitude loop commands g the aircraft cannot make: at the
    # bottom of the speed band it asked for 5.7 g, got 2.2, saturated the
    # elevator 30 % of the time and still drifted 192 ft.
    n_reserve: float = N_RESERVE   # never plan to use the last of the lift
    n_climb_max: float = 1.20  # cap on the g earmarked for altitude recovery


@dataclass
class AutopilotOutput:
    aileron: float
    elevator: float
    rudder: float
    throttle: float
    phi_cmd: float      # [rad] -- the target roll the heading loop asked for
    nz_cmd: float       # [g]
    max_bank: float     # [rad] -- ceiling that applied this step


class Autopilot:
    """One aircraft's inner loops.  Owns integrator state, so reset it."""

    def __init__(self, h0_ft: float = H0_FT, gains: AutopilotGains | None = None,
                 throttle_cap: float = THROTTLE_CAP) -> None:
        self.h0_ft = h0_ft
        self.g = gains or AutopilotGains()
        self.throttle_cap = throttle_cap
        self._i_nz = 0.0
        self._i_v = 0.0

    def reset(self) -> None:
        self._i_nz = 0.0
        self._i_v = 0.0

    def update(self, s, psi_cmd: float, v_cmd_kt: float, dt: float,
               alt_cmd_ft: float | None = None) -> AutopilotOutput:
        """s is an AircraftState.  psi_cmd rad, v_cmd_kt knots, alt_cmd_ft feet.

        `alt_cmd_ft` defaults to the altitude this autopilot was built with, so
        the locked-plane behaviour is what you get by not passing it.
        """
        g = self.g
        alt_cmd = self.h0_ft if alt_cmd_ft is None else _clip(
            alt_cmd_ft, ALT_MIN_FT, ALT_MAX_FT)

        # --- load-factor budget: altitude first, turn gets the remainder -----
        h_err_ft = alt_cmd - s.h_ft
        hdot_cmd = _clip(g.k_h * h_err_ft, -g.hdot_limit, g.hdot_limit)
        hdot_ft = s.h_dot / 0.3048
        n_climb = _clip(g.k_hd * (hdot_cmd - hdot_ft), 0.0, g.n_climb_max)

        n_avail = max(1.05, n_max(s.v_kt, s.h_ft) - g.n_reserve)
        n_turn = max(1.02, n_avail - n_climb)
        mb = math.acos(1.0 / n_turn)

        # --- heading -> target bank -> aileron ------------------------------
        psi_err = wrap_pi(psi_cmd - s.psi)
        phi_cmd = _clip(g.k_psi * psi_err - g.k_psi_r * s.r, -mb, mb)
        aileron = _clip1(g.k_phi * wrap_pi(phi_cmd - s.phi) - g.k_p * s.p)

        # --- altitude -> Nz -> elevator -------------------------------------
        # 1/cos(phi) is the load factor a level turn needs; without it the
        # aircraft descends the moment it banks.  Capped at what is available so
        # the inner loop never chases a target the wing cannot deliver.
        phi_ff = min(abs(s.phi), math.radians(g.phi_ff_limit_deg))
        nz_turn = 1.0 / max(math.cos(phi_ff), 0.10)
        nz_cmd = _clip(nz_turn + g.k_hd * (hdot_cmd - hdot_ft), -1.0, n_avail)

        err = nz_cmd - s.nz
        self._i_nz = _clip(self._i_nz + err * dt, -g.i_nz_limit, g.i_nz_limit)
        u = g.k_nz * err + g.k_nzi * self._i_nz
        if abs(u) > 1.0:                       # anti-windup: stop integrating
            self._i_nz -= err * dt             # once the stick is saturated
            u = _clip1(u)
        elevator = u                           # positive = pull (backend flips)

        # --- speed -> throttle ----------------------------------------------
        v_cmd_kt = _clip(v_cmd_kt, V_MIN_KT, V_MAX_KT)
        v_err = v_cmd_kt - s.v_kt
        self._i_v = _clip(self._i_v + v_err * dt, -g.i_v_limit, g.i_v_limit)
        thr_raw = g.thr_bias + g.k_v * v_err + g.k_vi * self._i_v
        throttle = _clip(thr_raw, 0.0, self.throttle_cap)
        if thr_raw != throttle:            # anti-windup on the throttle cap
            self._i_v -= v_err * dt

        return AutopilotOutput(aileron=aileron, elevator=elevator, rudder=0.0,
                              throttle=throttle, phi_cmd=phi_cmd,
                              nz_cmd=nz_cmd, max_bank=mb)


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _clip1(v: float) -> float:
    return _clip(v, -1.0, 1.0)
