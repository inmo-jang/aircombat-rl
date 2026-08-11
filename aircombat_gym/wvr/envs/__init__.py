"""The environments students train against, and the machinery under them.

    import gymnasium as gym
    import aircombat_gym.wvr.envs           # registers the ids

    env = gym.make("AirCombat/Circular-v0")

One file per assignment.  `base.py` holds what they share: `Combat` runs the two
aircraft and records what the weapon did, `TaskEnv` is the Gymnasium surface and
decides who won.  An assignment supplies its starting geometries, its opponent,
and -- if its difficulty needs it -- its own `verdict()`.

Every environment hands out the same 37-channel observation (`aircombat_gym.wvr.obs`)
so a policy trained on one rung loads into the next.  None of them computes a
reward: that is the student's job, and `tests/test_envs.py` enforces it.
"""
from __future__ import annotations

from gymnasium.envs.registration import register, registry

from .base import ARENA_R, Combat, Initial, TaskEnv
from .circular import CircularTargetEnv

ENVS = {"AirCombat/Circular-v0": CircularTargetEnv}


def _register() -> None:
    for env_id, cls in ENVS.items():
        if env_id in registry:            # importing twice must be harmless
            continue
        register(id=env_id,
                 entry_point=f"{cls.__module__}:{cls.__qualname__}",
                 # the assignment owns its own clock; Gymnasium's TimeLimit
                 # would truncate on a step count unrelated to it
                 max_episode_steps=None,
                 order_enforce=True)


_register()

__all__ = ["Combat", "Initial", "TaskEnv", "CircularTargetEnv", "ENVS",
           "ARENA_R"]
