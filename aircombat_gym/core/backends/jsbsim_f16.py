"""JSBSim F-16 backend.

Five things about JSBSim will silently produce wrong physics rather than an
error.  Each is handled below and each is marked TRAP-n; do not remove one
without reading docs/conversation_log.md section 5 first.

  TRAP-1  Neither run_ic() nor do_trim() starts the engine.  Measured with it
          off you get -1,351 lbf of windmilling drag and a glider.
  TRAP-2  run_ic() is not a reset.  FCS integrators survive it, so a reset
          after a hard pull answers the next full-aft stick with Nz = 0.99.
  TRAP-3  Attitude and rate ICs must be stated.  Otherwise an aircraft that
          ended inverted starts the next episode inverted, and "pull" is down.
  TRAP-4  position/distance-from-start-* is an unsigned distance.  Crossing the
          starting parallel reflects the coordinate.  Compute the local tangent
          plane from lat/long directly.
  TRAP-5  Euler angles are singular near the vertical: psi and phi can jump
          126 deg in a single 1/120 s step.  Guarded in AircraftState.nose_2d.

Also: throttle-cmd-norm 1.0 is full afterburner (throttle-pos-norm 2.0), not
military power, and FGFDMExec needs its root directory stated explicitly or a
second JSBSim installation can win.
"""
from __future__ import annotations

import math
import os

import jsbsim

from ..envelope import CENTER_LAT_DEG, CENTER_LON_DEG
from .base import FT, AircraftBackend, AircraftState

R_EARTH = 6371000.0
KT = 0.514444


class JSBSimF16(AircraftBackend):

    def __init__(self, dt_physics: float = 1.0 / 120.0,
                 h0_ft: float = 20000.0, constant_mass: bool = True) -> None:
        self._dt = dt_physics
        self._h0_ft = h0_ft
        self._constant_mass = constant_mass
        # TRAP: state the root directory; get_default_root_dir() may point elsewhere
        self._fdm = jsbsim.FGFDMExec(os.path.dirname(jsbsim.__file__), None)
        self._fdm.set_debug_level(0)
        self._fdm.load_model("f16")
        self._fdm.set_dt(dt_physics)
        self._lat0 = 0.0
        self._lon0 = 0.0
        self._trim_ok = False

    # -- lifecycle -----------------------------------------------------------

    def reset(self, x: float = 0.0, y: float = 0.0, psi: float = 0.0,
              v_kt: float = 400.0, h_ft: float | None = None) -> None:
        h_ft = self._h0_ft if h_ft is None else h_ft
        fdm = self._fdm

        # Origin of the local tangent plane.  Offsetting the *initial* lat/lon by
        # the requested (x, y) keeps the plane's origin at (0, 0) for both
        # aircraft, which is what the game geometry expects.
        lat0_deg, lon0_deg = CENTER_LAT_DEG, CENTER_LON_DEG
        self._lat0 = math.radians(lat0_deg)
        self._lon0 = math.radians(lon0_deg)
        lat = lat0_deg + math.degrees(y / R_EARTH)
        lon = lon0_deg + math.degrees(x / (R_EARTH * math.cos(self._lat0)))

        fdm["ic/h-sl-ft"] = h_ft
        fdm["ic/vt-kts"] = v_kt
        fdm["ic/psi-true-deg"] = math.degrees(psi) % 360.0
        # geodetic, not geocentric: ic/lat-gc-deg would place it 0.19 deg north
        fdm["ic/lat-geod-deg"] = lat
        fdm["ic/long-gc-deg"] = lon
        # TRAP-3: every attitude and rate, explicitly
        fdm["ic/phi-deg"] = 0.0
        fdm["ic/theta-deg"] = 0.0
        fdm["ic/alpha-deg"] = 0.0
        fdm["ic/beta-deg"] = 0.0
        fdm["ic/gamma-deg"] = 0.0
        fdm["ic/p-rad_sec"] = 0.0
        fdm["ic/q-rad_sec"] = 0.0
        fdm["ic/r-rad_sec"] = 0.0

        # TRAP-2: run_ic() alone leaves the FCS integrators holding whatever the
        # last episode ended with.  Zero the stick first so nothing is latched,
        # then reset_to_initial_conditions(0), which is the call that actually
        # reinitialises the control system.  Skipping this makes every episode
        # after the first depend on the one before it -- an H4 tournament that
        # reused one instance across 168 matches produced 119 draws and zero
        # kills purely from carried-over state, and the same match replayed
        # later gave a different winner.
        self.set_controls(0.0, 0.0, 0.0, 0.0)
        fdm.run_ic()
        fdm.reset_to_initial_conditions(0)
        # TRAP-1: the engine is off until you say so
        fdm["propulsion/engine[0]/set-running"] = 1
        fdm["fcs/throttle-cmd-norm"] = 1.0
        try:
            fdm.do_trim(1)          # 1 = level trim
            self._trim_ok = True
        except RuntimeError:
            # happens below roughly 180 kt, where level flight is impossible
            self._trim_ok = False
        self.set_controls(0.0, 0.0, 0.0, 0.5)

        if self._constant_mass:
            # D21: fuel burn would make the aircraft lighter every episode, so the
            # environment itself would drift during training.
            for tank in range(int(fdm["propulsion/total-fuel-lbs"] > 0) * 8):
                try:
                    fdm[f"propulsion/tank[{tank}]/external-flow-rate-pps"] = 0.0
                except Exception:
                    break
            fdm["propulsion/refuel"] = 1

    def hard_reset(self) -> None:
        """TRAP-2: the only reset that also clears FCS integrator state."""
        self._fdm.reset_to_initial_conditions(0)
        self._fdm["propulsion/engine[0]/set-running"] = 1
        self.set_controls(0.0, 0.0, 0.0, 0.5)

    # -- control -------------------------------------------------------------

    def set_controls(self, aileron: float, elevator: float,
                     rudder: float, throttle: float) -> None:
        fdm = self._fdm
        fdm["fcs/aileron-cmd-norm"] = _clip1(aileron)
        # measured: negative elevator-cmd-norm is nose-up, and 0 is exactly 1 g
        # in every flight condition (it is a g-command system).  Flip it so that
        # positive always means "pull" above this line.
        fdm["fcs/elevator-cmd-norm"] = -_clip1(elevator)
        # measured: positive rudder-cmd-norm yaws left.  Flip for the same reason.
        fdm["fcs/rudder-cmd-norm"] = -_clip1(rudder)
        fdm["fcs/throttle-cmd-norm"] = min(max(throttle, 0.0), 1.0)

    def run_one(self) -> None:
        self._fdm.run()

    # -- state ---------------------------------------------------------------

    @property
    def dt_physics(self) -> float:
        return self._dt

    @property
    def trim_ok(self) -> bool:
        return self._trim_ok

    @property
    def controls(self) -> dict:
        """What actually reached JSBSim, read back rather than recomputed.

        Two layers worth seeing: the *command* we wrote to fcs/*-cmd-norm, and
        the *surface* the FLCS chose in response.  They are not the same thing
        and the gap is often large -- the pitch channel is a g-command system,
        so a -0.52 elevator command lands as barely a degree of deflection once
        the FLCS has decided how much it actually needs.

        Note the rudder: it sits at ~12 deg with zero command.  That is the
        bundled f16's yaw channel, not us (JSBSim discussion #814).

        The three `pitch_*` entries open up the alpha limiter, which is
        otherwise invisible and explains most of "I am pulling full and nothing
        is happening".  The FLCS pitch channel sums three things and clips:

            pitch-scheduler = elevator-scheduler   pilot command, multiplied by
                                                   a gain that falls from 1.0 at
                                                   alpha 0 to 0.11 at 28.6 deg
                            + alpha-limiter-norm   alpha_rad * 1.0472, opposing
                            + g-load-pid           the g-command loop

        Measured at full aft stick, held 3 s at 20,000 ft: at 300 kt alpha
        settles at 15.8 deg, the schedule has cut pilot authority to 51 % and
        the limiter is pushing back 0.29, so only -0.12 of the -1.00 reaches the
        actuator.  At 600 kt alpha is 8.5 deg, authority 74 %, and the binding
        constraint is the g-command saturating instead.
        """
        fdm = self._fdm
        cmd = fdm["fcs/elevator-cmd-norm"]
        sched = fdm["fcs/elevator-scheduler"]
        return dict(
            aileron_cmd=fdm["fcs/aileron-cmd-norm"],
            elevator_cmd=cmd,
            rudder_cmd=fdm["fcs/rudder-cmd-norm"],
            throttle_cmd=fdm["fcs/throttle-cmd-norm"],
            throttle_pos=fdm["fcs/throttle-pos-norm"],
            aileron_deg=fdm["fcs/left-aileron-pos-deg"],
            elevator_deg=fdm["fcs/elevator-pos-deg"],
            rudder_deg=fdm["fcs/rudder-pos-deg"],
            # what the alpha schedule leaves of the pilot's pitch command, 0..1
            pitch_authority=abs(sched / cmd) if abs(cmd) > 1e-6 else 1.0,
            # how hard the limiter is pushing the other way, in command units
            pitch_limiter=fdm["fcs/alpha-limiter-norm"],
            # the sum of all three, clipped -- what the elevator actuator gets
            pitch_net=fdm["fcs/pitch-scheduler"],
            roll_rate_cmd=fdm["fcs/roll-rate-command"],
        )

    @property
    def state(self) -> AircraftState:
        fdm = self._fdm
        # TRAP-4: derive position from lat/long, never from distance-from-start-*
        # Geodetic on both ends.  Setting ic/lat-geod-deg and reading
        # position/lat-gc-rad mixes geodetic with geocentric and they differ by
        # 0.19 deg at this latitude -- a silent 20 km offset that put both
        # aircraft outside the arena on the first frame.
        lat = math.radians(fdm["position/lat-geod-deg"])
        lon = fdm["position/long-gc-rad"]
        x = (lon - self._lon0) * R_EARTH * math.cos(self._lat0)
        y = (lat - self._lat0) * R_EARTH

        vn = fdm["velocities/v-north-fps"] * FT
        ve = fdm["velocities/v-east-fps"] * FT

        return AircraftState(
            t=fdm["simulation/sim-time-sec"],
            x=x, y=y,
            h=fdm["position/h-sl-ft"] * FT,
            v=fdm["velocities/vtrue-fps"] * FT,
            h_dot=fdm["velocities/h-dot-fps"] * FT,
            psi=fdm["attitude/psi-rad"],
            theta=fdm["attitude/theta-rad"],
            phi=fdm["attitude/phi-rad"],
            p=fdm["velocities/p-rad_sec"],
            q=fdm["velocities/q-rad_sec"],
            r=fdm["velocities/r-rad_sec"],
            alpha=math.radians(fdm["aero/alpha-deg"]),
            beta=math.radians(fdm["aero/beta-deg"]),
            gamma=fdm["flight-path/gamma-rad"],
            nx=fdm["accelerations/n-pilot-x-norm"],
            ny=fdm["accelerations/n-pilot-y-norm"],
            # n-pilot-z-norm reads -1 in level flight
            nz=-fdm["accelerations/n-pilot-z-norm"],
            track=math.atan2(ve, vn),
            lat=fdm["position/lat-geod-deg"],
            lon=fdm["position/long-gc-deg"],
            thrust=fdm["propulsion/engine[0]/thrust-lbs"],
        )


def _clip1(u: float) -> float:
    return min(max(float(u), -1.0), 1.0)
