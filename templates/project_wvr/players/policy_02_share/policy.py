"""A shared policy, made by `tools.share` -- for flying matches only.

The same file in every shared copy.  `policy_net.zip` holds the network as a
list of layers; `wrappers.py` holds the State it reads.  The arithmetic is
torch's, as in training, so the copy picks exactly the actions the original did.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from wrappers import State

_ACT = {
    "relu": lambda x, a: torch.relu(x),
    "tanh": lambda x, a: torch.tanh(x),
    "sigmoid": lambda x, a: torch.sigmoid(x),
    "silu": lambda x, a: F.silu(x),
    "leaky_relu": lambda x, a: F.leaky_relu(x, a),
    "elu": lambda x, a: F.elu(x, a),
}


class Policy:

    ACTION_MODE = "continuous"

    def __init__(self, weights, device: str = "cpu"):
        self.state = State()
        with np.load(weights, allow_pickle=False) as z:
            self.head = str(z["head"])
            self.layers = []
            for i, kind in enumerate(str(k) for k in z["ops"]):
                if kind == "linear":
                    self.layers.append((kind, torch.from_numpy(z[f"w{i}"]),
                                        torch.from_numpy(z[f"b{i}"]), 0.0))
                else:
                    self.layers.append((kind, None, None, float(z[f"a{i}"])))

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(self.state(obs), dtype=torch.float32).reshape(1, -1)
        for kind, w, b, a in self.layers:
            x = F.linear(x, w, b) if kind == "linear" else _ACT[kind](x, a)
        x = x.reshape(-1).numpy()
        if self.head == "argmax":
            return int(np.argmax(x))
        return np.clip(x, -1.0, 1.0)

    def __str__(self) -> str:
        sizes = [w.shape[0] for kind, w, _, _ in self.layers if kind == "linear"]
        return f"shared {self.state.dim}->{sizes[:-1]}->{sizes[-1]}"
