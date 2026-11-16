"""Policy 

"""
from __future__ import annotations

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from wrappers import ACTION_MODE, State

# =============================================================================
# TODO 1 -- algorithm and network
# =============================================================================

NET_ARCH = [64, 64]            # layer widths
ACTIVATION = torch.nn.ReLU

# =============================================================================


class Policy:

    #: `discrete` or `continuous`, from `wrappers`.  `tools.grade` reads it here
    #: and builds the env the way training did.
    ACTION_MODE = ACTION_MODE

    #: default training budget
    TOTAL_STEPS = 200_000

    @staticmethod
    def make_learner(make_env, seed: int, device: str):
        """Build the learner.  `make_env` is a factory, not an env."""
        return DQN(
            "MlpPolicy", Monitor(make_env(seed)), seed=seed, device=device,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            tau=1.0,                         # soft target update
            target_update_interval=1_000,
            exploration_fraction=0.1,        # share of total steps spent decaying epsilon
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=NET_ARCH, activation_fn=ACTIVATION),
            verbose=0,
        )

    def __init__(self, weights=None, device: str = "cpu", model=None):
        """Wrap a live `model`, or load `weights` from disk."""
        self.state = State()
        self.model = model if model is not None else DQN.load(weights, device=device)

    def act(self, obs):
        """raw channels -> one action. """
        a, _ = self.model.predict(self.state(obs), deterministic=True)
        return int(a) if ACTION_MODE == "discrete" else a

    def __str__(self) -> str:
        """Algorithm and shape in one line: input -> hidden -> actions."""
        arch = self.model.policy.net_arch
        return f"DQN {self.state.dim}->{arch}->{self.model.action_space.n}"
