"""Scripted opponents for the gun task.  One so far, deliberately.

These sit in `wvr/` rather than at the package root because an opponent is
written against a task: a bot for the gun fight is one that turns and holds
energy, and a bot for a missile fight is one that notches and defends.  Nothing
about `Circler` transfers to a scenario where the weapon is different.

A bot here is anything with `reset()` and `act(state) -> (d_heading, d_speed,
d_alt)`, the same three deltas a policy emits.  It sees the aircraft state
rather than the observation vector, because these are the referee's aircraft,
not entrants.

`Circler` is the easiest target there is: it holds altitude and speed and turns
one way forever.  That is the point -- it is rung 1 of the difficulty ladder,
the case where anything that can point and close should score.  When it stops
being trivially catchable, something is wrong with the weapon or the geometry
rather than with the pilot.
"""
from __future__ import annotations

from .spaces import DELTA_ALT_FT, DELTA_HEADING_DEG, DELTA_SPEED_KT


class Circler:
    """Constant turn, constant height, constant speed.  No evasion, no fire.

    Speed and altitude hold themselves: guidance freezes a target when the delta
    is zero, so after the first step the only command it ever issues is a turn.
    Turning right or left is set once and never changes -- a target that
    reverses is rung 2, not this.
    """

    name = "circler"

    def __init__(self, turn_deg: float = DELTA_HEADING_DEG[-1],
                 right: bool = True) -> None:
        self.turn = turn_deg if right else -turn_deg

    def reset(self) -> None:
        pass

    def act(self, state) -> tuple[float, float, float]:
        return self.turn, DELTA_SPEED_KT[1], DELTA_ALT_FT[1]     # turn, hold, hold


ALL = (Circler,)
BY_NAME = {c.name: c for c in ALL}
