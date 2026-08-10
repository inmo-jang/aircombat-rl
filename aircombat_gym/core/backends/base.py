"""Backend contract.  Everything above this line is unaware of JSBSim.

The backend is deliberately *low level*: it takes control-surface commands and
advances physics.  Guidance lives in `control/`, one layer up, so the two can be
replaced independently -- swapping the guidance implementation (hand-written vs
LQR vs learned) must not require touching the simulator wrapper.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

FT = 0.3048


@dataclass(frozen=True)
class AircraftState:
    """Everything the layers above are allowed to see.  SI units, radians."""

    t: float          # sim time [s]
    x: float          # local tangent plane, east [m]
    y: float          # local tangent plane, north [m]
    h: float          # altitude MSL [m]
    v: float          # true airspeed [m/s]
    h_dot: float      # climb rate [m/s]
    psi: float        # heading (nose) [rad]
    theta: float      # pitch [rad]
    phi: float        # bank [rad]
    p: float          # roll rate [rad/s]
    q: float          # pitch rate [rad/s]
    r: float          # yaw rate [rad/s]
    alpha: float      # angle of attack [rad]
    beta: float       # sideslip [rad]
    gamma: float      # flight path angle [rad]
    nz: float         # load factor [g], +1 in level flight
    track: float      # velocity-vector ground track [rad]
    lat: float        # geodetic latitude [deg]  -- for TacView
    lon: float        # geodetic longitude [deg]
    thrust: float     # [lbf], diagnostic only

    @property
    def v_kt(self) -> float:
        return self.v / 0.514444

    @property
    def h_ft(self) -> float:
        return self.h / FT

    @property
    def nose_2d(self) -> tuple[float, float]:
        """Unit gun line projected onto the horizontal plane, (east, north).

        With altitude locked the aircraft is always near level, so this is the
        gun line to within a couple of degrees.  Guarded anyway: Euler angles
        are singular near the vertical and psi can jump 126 deg in one step.
        """
        return math.sin(self.psi), math.cos(self.psi)

    @property
    def nose_valid(self) -> bool:
        return abs(self.theta) < math.radians(60.0)

    @property
    def flyable(self) -> bool:
        """False once this has stopped being an aircraft.

        JSBSim does not fail gracefully on ground impact.  Measured, flying it
        in at 76 deg nose down: altitude went 3.9 m -> 3,861,096 m -> NaN over
        two decision steps.  Anything consuming states has to notice and stop
        rather than propagate that -- the NaN reached a renderer and took the
        tool down with it.
        """
        if not all(math.isfinite(v) for v in
                   (self.x, self.y, self.h, self.v, self.psi, self.theta,
                    self.phi, self.nz)):
            return False
        return 0.0 < self.h_ft < 100000.0


class AircraftBackend(ABC):
    """One aircraft's physics."""

    @abstractmethod
    def reset(self, x: float, y: float, psi: float, v_kt: float,
              h_ft: float | None = None) -> None:
        """Place the aircraft, trimmed and running, at a repeatable state."""

    @abstractmethod
    def set_controls(self, aileron: float, elevator: float,
                     rudder: float, throttle: float) -> None:
        """Stick and throttle, all normalised.

        Sign convention above this line is always "positive = right / pull";
        the backend flips whatever the simulator wants.
        """

    @abstractmethod
    def run_one(self) -> None:
        """Advance one physics step (dt_physics)."""

    @property
    @abstractmethod
    def state(self) -> AircraftState: ...

    @property
    @abstractmethod
    def dt_physics(self) -> float: ...
