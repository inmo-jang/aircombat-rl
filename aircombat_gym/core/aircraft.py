"""Backend + autopilot, stepped at the decision rate.

This is what the env and the flight-test tools both drive.  One `step()` is one
policy decision (20 Hz); the autopilot loops run underneath at physics rate,
because closing a bank loop at 20 Hz is not the same thing at all.

Command semantics, which are subtler than they look:

  delta != 0   target = current state + delta, recomputed every decision.  The
               target therefore stays just ahead of the aircraft for as long as
               the action is held, which makes it a *rate* command: hold "+30"
               and you turn at maximum performance indefinitely.  This is what
               LAG does, and why its increments are small.
  delta == 0   the target is *frozen*, not re-derived.  Re-deriving would set
               target = current, leaving zero error, so nothing would hold the
               aircraft against drift -- measured 400 -> 420 kt in 10 s of
               "hold".  Freezing makes action 0 mean what the spec says it
               means: keep this heading and this speed.
"""
from __future__ import annotations

import math

from .backends.base import AircraftState
from .backends.jsbsim_f16 import JSBSimF16
from .control.autopilot import Autopilot, AutopilotOutput, wrap_pi
from .envelope import ALT_MAX_FT, ALT_MIN_FT, H0_FT, V_MAX_KT, V_MIN_KT

# Physics is always 120 Hz.  The *decision* rate is a design choice, and it was
# fixed at 20 Hz before anything had been trained.  That is four times finer than
# LAG, which decides every 0.2 s, and it costs twice: each action changes the
# world a quarter as much, and a 90 s episode becomes 1,800 steps, so a 500k
# budget is only ~280 episodes.  Both hurt credit assignment.
PHYSICS_HZ = 120.0
DECISION_HZ = 20.0
SUBSTEPS = int(round(PHYSICS_HZ / DECISION_HZ))


class Aircraft:

    def __init__(self, h0_ft: float = H0_FT, autopilot: Autopilot | None = None,
                 backend: JSBSimF16 | None = None) -> None:
        self.h0_ft = h0_ft
        self.backend = backend or JSBSimF16(dt_physics=1.0 / PHYSICS_HZ, h0_ft=h0_ft)
        self.autopilot = autopilot or Autopilot(h0_ft=h0_ft)
        self.psi_cmd = 0.0
        self.v_cmd_kt = 400.0
        self.alt_cmd_ft = h0_ft
        self.last: AutopilotOutput | None = None

    def reset(self, x: float = 0.0, y: float = 0.0, psi: float = 0.0,
              v_kt: float = 400.0) -> AircraftState:
        self.backend.reset(x=x, y=y, psi=psi, v_kt=v_kt, h_ft=self.h0_ft)
        self.autopilot.reset()
        self.psi_cmd = psi
        self.v_cmd_kt = v_kt
        self.alt_cmd_ft = self.h0_ft
        self.last = None
        return self.state

    def step(self, delta_heading_deg: float = 0.0,
             delta_speed_kt: float = 0.0,
             delta_alt_ft: float = 0.0) -> AircraftState:
        """One decision step.  See the module docstring for what delta 0 means."""
        s = self.state
        if delta_heading_deg != 0.0:
            self.psi_cmd = wrap_pi(s.psi + math.radians(delta_heading_deg))
        if delta_speed_kt != 0.0:
            self.v_cmd_kt = min(max(s.v_kt + delta_speed_kt, V_MIN_KT), V_MAX_KT)
        if delta_alt_ft != 0.0:
            self.alt_cmd_ft = min(max(s.h_ft + delta_alt_ft,
                                      ALT_MIN_FT), ALT_MAX_FT)
        return self.hold()

    def nudge_target(self, d_heading_deg: float = 0.0,
                     d_speed_kt: float = 0.0, d_alt_ft: float = 0.0) -> None:
        """Move the targets themselves, autopilot-bug style.

        Different from `step`: the bug persists where you put it instead of
        being re-derived from the current state, so turn strength comes from how
        far ahead you set it.  Used by the keyboard tool; policies use `step`.
        """
        self.psi_cmd = wrap_pi(self.psi_cmd + math.radians(d_heading_deg))
        self.v_cmd_kt = min(max(self.v_cmd_kt + d_speed_kt, V_MIN_KT), V_MAX_KT)
        self.alt_cmd_ft = min(max(self.alt_cmd_ft + d_alt_ft,
                                  ALT_MIN_FT), ALT_MAX_FT)

    def hold(self, n_substeps: int | None = None) -> AircraftState:
        """Advance without changing the commands (used by step and by pause)."""
        dt = self.backend.dt_physics
        for _ in range(SUBSTEPS if n_substeps is None else n_substeps):
            s = self.backend.state
            out = self.autopilot.update(s, self.psi_cmd, self.v_cmd_kt, dt,
                                       self.alt_cmd_ft)
            self.backend.set_controls(out.aileron, out.elevator,
                                      out.rudder, out.throttle)
            self.backend.run_one()
            self.last = out
        return self.state

    @property
    def state(self) -> AircraftState:
        return self.backend.state
