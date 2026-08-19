"""Which assignment am I working on, and which environment is that?

    python -m tools.grade solutions/project_01_circular
    python -m tools.watch templates/project_01_circular --tacview

Every tool takes a **path** to the assignment folder as its first argument.  A
bare name would not do: the same assignment exists under `solutions/` and under
`templates/`, and a name that resolves to one of them by convention is a command
that keeps running after a move while measuring something else.

**The whole per-assignment configuration is the table below.**  Folder names
cannot supply the environment: `project_03_advantaged` runs `AdvantagedFight`,
so the convention breaks exactly where guessing would.

Everything else falls out:

    TITLE   from the number in the folder name
    DUEL    `issubclass(ENV, DuelEnv)` -- can the learner be shot down?  That
            decides whether seats are swapped when scoring, and asking the class
            is better than a hand-set flag that can disagree with it.

A submission never contains any of this.  The assignment is the professor's
side; the submission is `policy.py` + `wrappers.py` + weights (`tools.policies`).
"""
from __future__ import annotations

import pathlib
import sys

#: engagements per score.  Fixed, so two scores can be read side by side.
STANDARD_N = 40

#: Seeds `TEST_BAND + 0..n-1` are what `--band` defaults to.  It is a **practice
#: band**: it is readable here, so anyone can evaluate against it repeatedly and
#: it is not held out from anyone.  Real marking passes `--band` with a band that
#: is not in this repository.
TEST_BAND = 700_000

#: folder name -> the registered environment id it is set on.  Keyed by name, not
#: by path, because the environment belongs to the assignment rather than to the
#: tree the assignment sits in.
ASSIGNMENTS = {
    "project_01_circular": "AirCombat/Circular-v0",
    "project_01_circular_custom": "AirCombat/Circular-v0",
    "project_02_evader": "AirCombat/Evader-v0",
    "project_03_advantaged": "AirCombat/AdvantagedFight-v0",
    "project_04_fair": "AirCombat/FairFight-v0",
}


class Project:
    """One assignment: where it lives, what it is called, what it runs on."""

    def __init__(self, folder: pathlib.Path, env_id: str):
        import aircombat_gym.wvr.envs as E
        from aircombat_gym.wvr.envs.base import DuelEnv

        self.DIR = folder
        self.ENV_ID = env_id
        self.ENV = E.ENVS[env_id]
        # Ask the class rather than trust a flag: a duel is exactly an
        # environment where the learner can lose, and that is what `DuelEnv` is.
        self.DUEL = issubclass(self.ENV, DuelEnv)
        self.TITLE = f"Assignment {folder.name.split('_')[1]}"

    def __repr__(self) -> str:
        return f"<{self.DIR.name} {self.ENV_ID} duel={self.DUEL}>"


def resolve(name: str) -> pathlib.Path:
    """A path to an assignment folder -> that folder.

    The path is kept, not turned back into a name: two trees hold a folder of
    the same name, and rebuilding the path from the name is how a command ends
    up measuring the other one.
    """
    p = pathlib.Path(name).resolve()
    if not p.is_dir():
        raise SystemExit(f"no such folder: {name}")
    if p.name not in ASSIGNMENTS:
        raise SystemExit(f"{p.name} is not an assignment.  "
                         f"Have: {', '.join(ASSIGNMENTS)}")
    return p


def load(name: str) -> Project:
    """The assignment, with its directory first on `sys.path`.

    The path insert is what lets a tool then say `import wrappers` and get
    *that* assignment's answer rather than the previous one's.
    """
    folder = resolve(name)
    proj = Project(folder, ASSIGNMENTS[folder.name])
    sys.path.insert(0, str(proj.DIR))
    # Every assignment names its modules the same, so a second `load` in one
    # interpreter has to evict the first or it silently uses the wrong code.
    for stale in ("policy", "wrappers", "utils", "train"):
        sys.modules.pop(stale, None)
    return proj


def design_of(proj: Project) -> pathlib.Path:
    """The assignment's own design, used when `--design` is not given.

    Any `baseline*/` holding a `policy.py`.  The name is not fixed -- an
    assignment may carry `baseline_dqn/` and `baseline_ppo/` side by side --
    but **exactly one of them, or the caller has to say which.**  Two designs
    and no flag is a score whose source cannot be read off the command.
    """
    found = sorted(d for d in proj.DIR.glob("baseline*")
                   if (d / "policy.py").exists())
    if len(found) == 1:
        return found[0]
    if not found:
        return proj.DIR
    raise SystemExit(
        f"{proj.DIR.name} holds more than one design "
        f"({', '.join(d.name for d in found)}).  Say which with --design.")


def design_path(proj: Project, given: str | None) -> pathlib.Path:
    """`--design` -> a folder.  Without it, the assignment's own `baseline*/`.

    A relative path is tried inside the assignment first, then against the
    working directory.
    """
    if given is None:
        return design_of(proj)
    p = pathlib.Path(given)
    if p.is_absolute():
        return p
    inside = proj.DIR / p
    if inside.is_dir():
        return inside
    if p.is_dir():
        return p.resolve()
    raise SystemExit(f"no such design {given!r} -- tried {inside} "
                     f"and {p.resolve()}")


def split_argv(argv: list[str] | None) -> tuple[str, list[str]]:
    """Peel the assignment off the front, leaving the tool its own arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        raise SystemExit("first argument is a path to the assignment, "
                         "e.g. `solutions/project_01_circular`")
    return argv[0], argv[1:]
