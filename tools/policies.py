"""Load a design and get something that flies.  Shared by `grade` and `watch`.

**A design folder is three files.**

    <design>/
        policy.py                defines `Policy`     <- network and how to run it
        wrappers.py              defines `State`      <- what the policy sees
        policy_net.pth           the parameters

and the grader touches exactly one symbol:

    from policy import Policy
    p = Policy("policy_net.pth")
    action = p.act(obs)                       # obs is the 39 raw channels

**Nothing here knows what is inside** -- not the algorithm, not the layer names,
not whether torch is involved.  The contract is *a policy that runs*, not the
shape of a weight file, so widening the network or bringing PPO costs the
grader nothing.

The reward is not in the contract: greedy replay never calls one.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

#: Named the same in every design, so a second `load` in one interpreter has to
#: evict the first or it silently grades the wrong code.
_SUBMISSION_MODULES = ("policy", "wrappers", "utils", "train")


class Mismatch(Exception):
    """The design cannot be run, with a sentence saying why."""


#: Weight file names.  Hand-written torch saves `.pth`, SB3 `model.save` writes
#: `.zip`.  Which one it is, is the design's `Policy` to open -- this only finds it.
WEIGHTS = ("policy_net.zip", "policy_net.pth")


def default_weights(design: pathlib.Path) -> pathlib.Path:
    """The design's weights."""
    for name in WEIGHTS:
        p = design / name
        if p.exists():
            return p
    raise Mismatch(f"no {' or '.join(WEIGHTS)} in {design} -- or pass --policy")


def load(design: pathlib.Path, weights: pathlib.Path, device: str = "cpu"):
    """(design, weights) -> (act, note).

    `act(obs)` is the callable the environment is stepped with; `note` is the
    one-line description a table or a viewer caption prints.

    Importing the design runs its code -- unavoidable, it *is* what is being
    graded -- so marking a batch should give each one its own process.
    """
    design = design.resolve()
    # Before importing, not after: `sys.path` accumulates and every design
    # names its modules the same, so a design with no `policy.py` would import
    # the previous one and be graded on somebody else's work.
    if not (design / "policy.py").exists():
        raise Mismatch(f"{design}/policy.py not found")

    sys.path.insert(0, str(design))
    for stale in _SUBMISSION_MODULES:
        sys.modules.pop(stale, None)
    try:
        mod = importlib.import_module("policy")
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", "policy")
        raise Mismatch(f"{design}/policy.py imports {missing!r}, "
                       f"which is not in the design") from None
    except Exception as e:                       # their code, their traceback
        raise Mismatch(f"{design}/policy.py failed to import: "
                       f"{type(e).__name__}: {e}") from None
    finally:
        # Leave the path as we found it, for the same reason.
        try:
            sys.path.remove(str(design))
        except ValueError:
            pass

    got = pathlib.Path(getattr(mod, "__file__", "")).resolve().parent
    if got != design:
        raise Mismatch(f"imported policy.py from {got}, not from {design}.  "
                       f"Refusing to grade the wrong design")

    if not hasattr(mod, "Policy"):
        raise Mismatch(f"{design}/policy.py defines no `Policy`.  "
                       f"The contract is one class with `.act(obs)`")
    try:
        policy = mod.Policy(str(weights), device=device)
    except TypeError:
        # `Policy(weights)` without the device argument is a fair reading of
        # the contract, so take it.
        try:
            policy = mod.Policy(str(weights))
        except Exception as e:
            raise Mismatch(f"Policy(weights) raised {type(e).__name__}: {e}") from None
    except RuntimeError as e:
        # The most common failure by far, worth naming rather than relaying
        # torch's version of it.
        raise Mismatch(
            f"the weights do not fit this Policy -- usually the observation "
            f"or the network changed after training.  torch said: {e}"
        ) from None
    except Exception as e:
        raise Mismatch(f"Policy(weights) raised {type(e).__name__}: {e}") from None

    if not hasattr(policy, "act"):
        raise Mismatch("Policy has no `.act(obs)`")
    return policy.act, str(policy) if _has_str(mod.Policy) else "Policy"


def _has_str(cls) -> bool:
    return cls.__str__ is not object.__str__
