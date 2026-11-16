"""Make a copy of a design that can fly a match and nothing else.

    python -m tools.share templates/project_wvr/2021130000_InmoJang
    python -m tools.share <design> --out <folder>

The copy holds the same three files as any design, so everything that loads a
design -- `match.py`, `tools.grade`, `tools.watch` -- loads it unchanged:

    policy.py         the same file in every shared copy: it runs the network
    wrappers.py       the design's `State` and what `State` needs.  Nothing else
    policy_net.zip    the network as plain arrays: layer weights and activations

**Left out:** the reward, the training code, the hyperparameters, which
algorithm trained it, every comment and docstring.  **Not hidden**, because a
match cannot run without it: the `State`, the size of each layer, and how the
policy flies.

The network is read out of the design's `Policy.model` -- a stable-baselines3
DQN, PPO, A2C, SAC or TD3 with the default MLP policy -- as the layers that
pick its greedy action.  So the copy only works for a `Policy.act` that is
`model.predict(state(obs), deterministic=True)`; one that switches between
networks, or adds logic of its own, is refused.  Before anything is written
the copy is loaded and flown against the original on real observations, and
it must pick the same action on every one.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import shutil
import sys
import tempfile

import numpy as np

from . import policies


class ShareError(Exception):
    """The design cannot be shared this way, with a sentence saying why."""


# =============================================================================
# the network -> a list of layers
# =============================================================================

def _layers(module) -> list[tuple]:
    """A torch module -> [("linear", W, b) | (activation, param), ...]."""
    import torch.nn as nn
    if isinstance(module, nn.Sequential):
        out = []
        for child in module:
            out += _layers(child)
        return out
    if isinstance(module, nn.Linear):
        w = module.weight.detach().cpu().numpy().astype(np.float32)
        b = (module.bias.detach().cpu().numpy().astype(np.float32)
             if module.bias is not None else np.zeros(w.shape[0], np.float32))
        return [("linear", w, b)]
    if isinstance(module, (nn.Identity, nn.Flatten, nn.Dropout)):
        return []
    simple = {nn.ReLU: "relu", nn.Tanh: "tanh", nn.Sigmoid: "sigmoid",
              nn.SiLU: "silu"}
    for cls, name in simple.items():
        if isinstance(module, cls):
            return [(name, 0.0)]
    if isinstance(module, nn.LeakyReLU):
        return [("leaky_relu", float(module.negative_slope))]
    if isinstance(module, nn.ELU):
        return [("elu", float(module.alpha))]
    raise ShareError(f"layer {type(module).__name__} is not supported")


def _flat(fx) -> None:
    if type(fx).__name__ != "FlattenExtractor":
        raise ShareError(f"features extractor {type(fx).__name__} is not "
                         f"supported; only the default MLP policy is")


def network(model) -> tuple[list[tuple], str]:
    """An SB3 model -> (layers, head).  `head` is `argmax` or `clip`."""
    p = getattr(model, "policy", None)
    space = getattr(model, "action_space", None)
    if p is None or space is None:
        raise ShareError("Policy.model is not a stable-baselines3 model")
    discrete = hasattr(space, "n")
    if not discrete and not (np.all(space.low == -1.0) and np.all(space.high == 1.0)):
        raise ShareError(f"action space {space} is not Box(-1, 1)")
    if getattr(p, "use_sde", False):
        raise ShareError("gSDE (use_sde=True) is not supported")

    if hasattr(p, "q_net") and hasattr(p.q_net, "q_net"):            # DQN
        _flat(p.q_net.features_extractor)
        return _layers(p.q_net.q_net), "argmax"
    if hasattr(p, "mlp_extractor") and hasattr(p, "action_net"):     # PPO, A2C
        _flat(p.features_extractor if p.share_features_extractor
              else p.pi_features_extractor)
        layers = _layers(p.mlp_extractor.policy_net) + _layers(p.action_net)
        return layers, ("argmax" if discrete else "clip")
    actor = getattr(p, "actor", None)
    if actor is not None and hasattr(actor, "latent_pi"):            # SAC
        _flat(actor.features_extractor)
        return (_layers(actor.latent_pi) + _layers(actor.mu)
                + [("tanh", 0.0)]), "clip"
    if actor is not None and hasattr(actor, "mu"):                   # TD3
        _flat(actor.features_extractor)
        return _layers(actor.mu), "clip"
    raise ShareError(f"{type(p).__name__} is not supported")


def save_network(path: pathlib.Path, layers, head: str) -> None:
    """Plain arrays in a zip -- `.npz` under the name the loaders look for."""
    arrays = {"ops": np.array([k for k, *_ in layers]), "head": np.array(head)}
    for i, (kind, *rest) in enumerate(layers):
        if kind == "linear":
            arrays[f"w{i}"], arrays[f"b{i}"] = rest
        else:
            arrays[f"a{i}"] = np.float32(rest[0])
    with open(path, "wb") as f:                # a file object: no `.npz` added
        np.savez(f, **arrays)


POLICY_PY = '''"""A shared policy, made by `tools.share` -- for flying matches only.

The same file in every shared copy.  `policy_net.zip` holds the network as a
list of layers; `wrappers.py` holds the State it reads.  The arithmetic is
torch's, as in training, so the copy picks exactly the actions the original did.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from wrappers import __STATE__

_ACT = {
    "relu": lambda x, a: torch.relu(x),
    "tanh": lambda x, a: torch.tanh(x),
    "sigmoid": lambda x, a: torch.sigmoid(x),
    "silu": lambda x, a: F.silu(x),
    "leaky_relu": lambda x, a: F.leaky_relu(x, a),
    "elu": lambda x, a: F.elu(x, a),
}


class Policy:

    ACTION_MODE = "__MODE__"

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
'''


# =============================================================================
# the State -> only the code it needs
# =============================================================================

_SCALAR = (bool, int, float, str)


def _bound(node) -> list[str]:
    """Top-level names a statement defines."""
    if isinstance(node, ast.Import):
        return [a.asname or a.name.split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    targets = (node.targets if isinstance(node, ast.Assign)
               else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign))
               else [])
    out = []
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.append(n.id)
    return out


def _used(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _no_docstrings(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = n.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            n.body = body or [ast.Pass()]
    return node


def trim(source: str, root: str, values: dict,
         design: pathlib.Path) -> tuple[str, list[str]]:
    """The statements `root` needs, and no others.

    A constant whose value is a plain number or string is written out as that
    value, so whatever it was computed from -- the task it was trained on, a
    module it was imported from -- is not carried along.
    """
    body = ast.parse(source).body
    defines: dict[str, list[int]] = {}
    for i, node in enumerate(body):
        for name in _bound(node):
            defines.setdefault(name, []).append(i)

    keep: set[int] = set()
    literal: dict[int, list[str]] = {}         # statement -> names written as values
    seen, todo = set(), [root]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for i in defines.get(name, []):
            node = body[i]
            inline = (not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                            ast.ClassDef, ast.Import))
                      and isinstance(values.get(name), _SCALAR))
            if inline:
                literal.setdefault(i, []).append(name)
            elif i not in keep:
                keep.add(i)
                todo.extend(_used(node))

    out, names = [], []
    for i, node in enumerate(body):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            out.append(node)
            continue
        if i in keep:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                top = (node.module or "") if isinstance(node, ast.ImportFrom) \
                    else node.names[0].name
                top = top.split(".")[0]
                if (design / f"{top}.py").exists() or (design / top).is_dir():
                    raise ShareError(f"State needs `{top}` from the design "
                                     f"folder; move what it uses into the "
                                     f"file that defines State")
            out.append(_no_docstrings(node))
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                names += _bound(node)
        for name in literal.get(i, []):
            out.append(ast.Assign(targets=[ast.Name(name, ast.Store())],
                                  value=ast.Constant(values[name]), lineno=0))
            names.append(name)
    head = '"""A shared copy, made by `tools.share` -- the State only."""\n'
    text = head + ast.unparse(ast.fix_missing_locations(
        ast.Module(body=out, type_ignores=[]))) + "\n"
    return text, names


# =============================================================================
# check, and the command
# =============================================================================

def check(act_a, act_b, mode: str, engagements: int) -> int:
    """Fly `act_a` against `Ace`; `act_b` must pick the same action every step."""
    from aircombat_gym.wvr.envs.fair import FairFightEnv
    seen = 0
    for k in range(engagements):
        env = FairFightEnv(action_mode=mode, seat=("red", "blue")[k % 2])
        obs, _ = env.reset(seed=k)
        while True:
            a, b = act_a(obs), act_b(obs)
            same = (a == b) if mode == "discrete" else np.array_equal(
                np.asarray(a, np.float32), np.asarray(b, np.float32))
            if not same:
                raise ShareError(
                    f"the copy picks {b!r} where the design picks {a!r} "
                    f"(observation {seen + 1}).  Does Policy.act do more than "
                    f"model.predict(state(obs), deterministic=True)?")
            seen += 1
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return seen


def share(design: pathlib.Path, out: pathlib.Path, *, force=False,
          engagements=2) -> None:
    design, out = design.resolve(), out.resolve()
    if out == design:
        raise ShareError("--out is the design itself")
    if out.exists() and any(out.iterdir()) and not force:
        raise ShareError(f"{out} exists; pass --force to replace it")

    act, note, mode = policies.load(design, policies.default_weights(design))
    policy = getattr(act, "__self__", None)
    model, state = getattr(policy, "model", None), getattr(policy, "state", None)
    if model is None or state is None:
        raise ShareError("Policy needs `self.model` (the SB3 model) and "
                         "`self.state` (the State), as the template has")
    layers, head = network(model)

    cls = type(state)
    fn = next((v for v in vars(cls).values() if hasattr(v, "__globals__")), None)
    if fn is None:
        raise ShareError(f"{cls.__name__} defines no methods to read")
    values = fn.__globals__
    src = pathlib.Path(values.get("__file__", ""))
    if src.parent.resolve() != design:
        raise ShareError(f"{cls.__name__} is defined outside the design folder")
    wrappers, kept = trim(src.read_text(encoding="utf-8"), cls.__name__,
                          values, design)

    # Built and checked outside the working tree, so a refused copy leaves
    # nothing behind and a synced folder never sees the half-made one.
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="share_")) / out.name
    tmp.mkdir()
    try:
        state_import = cls.__name__ if cls.__name__ == "State" \
            else f"{cls.__name__} as State"
        (tmp / "policy.py").write_text(
            POLICY_PY.replace("__STATE__", state_import).replace("__MODE__", mode),
            encoding="utf-8")
        (tmp / "wrappers.py").write_text(wrappers, encoding="utf-8")
        save_network(tmp / "policy_net.zip", layers, head)

        copy_act, copy_note, copy_mode = policies.load(tmp, tmp / "policy_net.zip")
        if copy_mode != mode:
            raise ShareError(f"the copy flies {copy_mode}, the design {mode}")
        n = check(act, copy_act, mode, engagements)

        # Files, not a directory rename: on Windows a synced folder (OneDrive)
        # or an open editor can hold a directory, and then neither rename nor
        # rmtree can take it.  What cannot be removed is only a stale cache.
        out.mkdir(parents=True, exist_ok=True)
        for f in out.iterdir():
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink()
        for name in ("policy.py", "wrappers.py", "policy_net.zip"):
            shutil.copy2(tmp / name, out / name)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    print(f"  {design.name}  {note}")
    print(f"  ->  {out}")
    print(f"      {copy_note}, {mode}")
    print(f"      wrappers.py keeps: {', '.join(kept)}")
    print(f"      same action on {n}/{n} observations")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.share", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", help="a folder with policy.py, wrappers.py and the weights")
    ap.add_argument("--out", default=None,
                    help="where the copy goes.  Default: share/<name> next to the design")
    ap.add_argument("--force", action="store_true", help="replace --out if it exists")
    ap.add_argument("--check", type=int, default=2, metavar="N",
                    help="engagements flown to compare copy and design (default 2)")
    args = ap.parse_args(argv)

    design = pathlib.Path(args.design)
    out = (pathlib.Path(args.out) if args.out
           else design.resolve().parent / "share" / design.resolve().name)
    try:
        share(design, out, force=args.force, engagements=args.check)
    except (ShareError, policies.Mismatch) as e:
        print(f"cannot share {design}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
