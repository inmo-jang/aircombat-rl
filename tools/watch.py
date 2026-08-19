"""Watch a policy fly the task it was trained on.

    python -m tools.watch solutions/project_01_circular
    python -m tools.watch solutions/project_01_circular --tacview     # live TacView
    python -m tools.watch solutions/project_01_circular --acmi out.acmi

    BACKSPACE  new episode      SPACE  pause
    TAB        replay speed     ESC    quit

The window comes from `aircombat_gym.tools.auto_operation.fly`, which takes an
env and an `act(obs)`.  It drives the task environment -- same weapon, same
clock, same opponent as grading, so what is on screen is what `tools.grade`
scores.
"""
from __future__ import annotations

import argparse
import pathlib

from . import policies, project as P


def main(argv=None) -> int:
    name, rest = P.split_argv(argv)
    proj = P.load(name)

    ap = argparse.ArgumentParser(prog=f"tools.watch {name}", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default=None,
                    help="a folder with policy.py, wrappers.py and the weights.  "
                         "Default: the assignment's own `baseline*/`")
    ap.add_argument("--policy", default=None,
                    help="a .pth or .zip.  Default: <design>/policy_net.*")
    ap.add_argument("--seed", type=int, default=None,
                    help="fixes the sequence of engagements")
    ap.add_argument("--acmi", metavar="PATH",
                    help="record to .acmi.  A relative path lands in the design folder")
    ap.add_argument("--tacview", action="store_true",
                    help="serve live telemetry (TacView Advanced, Windows)")
    ap.add_argument("--tacview-port", type=int, default=42674)
    args = ap.parse_args(rest)

    design = P.design_path(proj, args.design)
    weights = (pathlib.Path(args.policy) if args.policy
               else policies.default_weights(design))

    try:
        act, note, mode = policies.load(design, weights)
    except policies.Mismatch as e:
        raise SystemExit(f"cannot fly this policy: {e}")
    env = proj.ENV(action_mode=mode)

    score = _known_score(design, weights)
    print(f"  {proj.TITLE} · {design.name}")
    print(f"  policy: {weights}")
    print(f"          {note}" + (f"   ({score})" if score else ""))

    sinks = []
    if args.acmi:
        from aircombat_gym.core.tacview import AcmiFile
        acmi = pathlib.Path(args.acmi)
        if not acmi.is_absolute():
            acmi = design / acmi
        acmi.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(AcmiFile(str(acmi)))
        print(f"  recording {acmi}")
    if args.tacview:
        from aircombat_gym.core.tacview import AcmiRealtime
        live = AcmiRealtime(port=args.tacview_port)
        sinks.append(live)
        print(f"  TacView real-time telemetry: {live.address}")
        print("    TacView: Record -> Real-time Telemetry -> that address")

    from aircombat_gym.tools.auto_operation import fly
    kills, n = fly(env, act, title=f"{proj.TITLE}  -  {design.name}",
                   subtitle=f"{note}   {score or ''}".strip(),
                   seed=args.seed, sinks=sinks)
    for s in sinks:
        s.close()
    print(f"\n{kills}/{n} kills over {n} episodes")
    return 0


def _known_score(design: pathlib.Path, weights: pathlib.Path) -> str:
    """The score `tools.grade` left, if any.

    Only the one beside these weights -- a score from before someone swapped the
    checkpoint is worse than none.
    """
    import json
    f = weights.parent / "score.json"
    if not f.exists():
        return ""
    try:
        m = json.loads(f.read_text(encoding="utf-8"))["rows"][0]["summary"]
        return f"{m['kills']}/{m['n']} = {m['rate']:.2f}"
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
