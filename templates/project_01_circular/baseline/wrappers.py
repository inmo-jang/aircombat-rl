"""Environment wrapping.

    State           39 raw channels -> policy input
    RewardFunction  what one step pays
    Shaped          the env wearing both
    make_env        builds the task env, for training or for evaluation
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from aircombat_gym.wvr import obs as O
from aircombat_gym.wvr.engagement import MUZZLE_MS
from aircombat_gym.wvr.envs.circular import CircularTargetEnv as TASK

# =============================================================================
# Task spec
# =============================================================================

WEZ_CONE_DEG = TASK.wez_cone_deg                        # half-angle of the firing cone [deg]
WEZ_R_MIN, WEZ_R_MAX = TASK.wez_r_min, TASK.wez_r_max   # effective range [m]
TRACK_LOCK_S = TASK.track_lock                          # unbroken track before damage starts [s]
T_MAX_S = TASK.t_max                                    # engagement clock [s]

OMEGA_MAX_RAD_S = math.radians(12.0)                    # sustainable turn rate [rad/s]

#: What the policy hands the environment: `discrete` or `continuous`.  The env
#: offers both; `tools.grade` builds it from `Policy.ACTION_MODE`.
ACTION_MODE = "discrete"


# =============================================================================
# TODO 2 -- state design
# =============================================================================

def _wrap(a: float) -> float:
    """Angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class State:
    """39 raw channels -> the vector the policy reads."""

    def __init__(self):
        self.dim = len(self.channels(np.zeros(O.STATE_DIM, dtype=np.float32)))

    def channels(self, obs) -> dict[str, float]:
        # The 39 channels the env hands over.  Raw SI, no normalisation, no
        # clipping, no wrapping.  15 per aircraft (`own_` then `opp_`), then 9
        # engagement channels -- the first four of those are `own_`/`opp_`
        # pairs, `t_remaining` is single.
        #
        #   x   y   h            position      [m]     east, north, MSL
        #   vx  vy  vz           velocity      [m/s]   east, north, up
        #   nx  ny  nz           load factor   [g]     body axes; level flight is nz=+1
        #   phi theta psi        attitude      [rad]   bank, pitch, heading
        #   p   q   r            body rates    [rad/s]
        #
        #   health              1.0 down to 0
        #   track_time          [s] unbroken track; resets to 0 when it breaks
        #   in_wez              0.0 / 1.0, inside the firing envelope
        #   dist_to_boundary    [m] to the arena edge
        #   t_remaining         [s]
        d = O.unpack(obs)

        # --- relative position and velocity --------------------------------
        dx = d["opp_x"] - d["own_x"]
        dy = d["opp_y"] - d["own_y"]
        dz = d["opp_h"] - d["own_h"]
        r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9

        # TODO: derive whatever else the policy needs.  Angles wrap, so feed
        # them as sin/cos pairs rather than radians, and keep every channel
        # roughly in [-1, 1].  `_wrap` and `MUZZLE_MS` are here for that.

        # --- channels.  Two to start with -----------------------------------
        return {
            "range":      math.exp(-r / WEZ_R_MAX),
            "own_health": d["own_health"],
        }

    def __call__(self, obs) -> np.ndarray:
        """`channels()` flattened for the network."""
        ch = self.channels(obs)
        return np.fromiter(ch.values(), dtype=np.float32, count=len(ch))


# =============================================================================
# TODO 3 -- reward design
# =============================================================================

class RewardFunction:
    """Reward for one step.

        info   engagement state: `range`, `ata_lead`, `aa`, `track_time`,
               `in_wez`, `own_health`, `opp_health`, `opp_speed`, `t`
        prev   the previous step's `info`, None on the first step
        won    None until the engagement ends
    """

    def __call__(self, info, prev, won) -> float:
        # A win is one reward in ~2,400 steps and never arrives by accident.
        # TODO: add terms that pay on the way there.
        return self._terminal(won)

    # --- terms ---------------------------------------------------------------

    def _terminal(self, won) -> float:
        """End of engagement only.  Win +1, loss -0.2."""
        if won is None:
            return 0.0
        return 1.0 if won else -0.2


# =============================================================================
# Building the env
# =============================================================================

class Shaped(gym.Wrapper):
    """The env wearing `State` and `RewardFunction`."""

    def __init__(self, env, state, reward_fn):
        super().__init__(env)
        self.state = state
        self.fn = reward_fn
        self._prev = None       # previous step's info, for the reward
        # advertise the processed width, not the raw 39
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(state.dim,), dtype=np.float32)

    def _act(self, action):
        """A `Discrete` env wants a python int; a `Box` one wants the array."""
        return action if ACTION_MODE == "continuous" else int(action)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._prev = None
        return self.state(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(self._act(action))
        reward = float(self.fn(info, self._prev, info.get("won")))
        self._prev = dict(info)
        return self.state(obs), reward, terminated, truncated, info


def make_env(seed: int | None = None, shaped: bool = True) -> gym.Env:
    """Build the task env.

        shaped=True   state and reward applied, ready to train on
        shaped=False  the 39 raw channels, untouched
        seed          seeds the env's own generator
    """
    env = TASK(action_mode=ACTION_MODE, seed=seed)
    return Shaped(env, State(), RewardFunction()) if shaped else env
