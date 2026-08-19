"""The standard score.  One design, one number, the same rules every time.

    python -m tools.grade solutions/project_01_circular
    python -m tools.grade solutions/project_01_circular --design subs/2021001
    python -m tools.grade solutions/project_01_circular --sweep results

The grader touches one symbol: `Policy(weights).act(obs)`.  It knows neither the
algorithm nor the network; the contract is in `tools.policies`.

**The rules, the same for every assignment.**

    band     `TEST_BAND` + 0..n-1.  Never seen in training, never used to pick a
             snapshot -- `val` does that, and scoring on `val` reads about 2x
             optimistic.  Real marking passes `--band` with a band that is not
             in this repository.
    n        40 engagements, so two scores can be read side by side.
    seats    red only, unless the assignment is in `SYMMETRIC_SEATS` -- then
             the 40 split 20 per seat, as a control on position.  A seat bias
             has turned up here before.
    policy   greedy.

**Compare standard scores only with each other.**  The same weights read 0.94 at
n=50 and 0.88 at n=40.  40 separates 0.88 from 0.28, the gap the ladder is built
on; it does not separate 0.88 from 0.94, and no honest n here would.
"""
from __future__ import annotations

import argparse
import json
import pathlib

from . import policies, project as P
from .project import STANDARD_N, TEST_BAND


def play(env, act, seed0: int, n: int, seat: str = "red") -> list[dict]:
    """`n` greedy engagements from `seed0`, one row each.

    Not only how each ended but how it went.  On kill rate alone a near miss and
    never having aimed read as the same 0, and they are different failures.
    """
    rows = []
    for i in range(n):
        seed = seed0 + i
        obs, info = env.reset(seed=seed)
        c, steps, longest = env._combat, 0, 0.0
        while True:
            obs, _, term, trunc, info = env.step(act(obs))
            steps += 1
            longest = max(longest, info["track_time"])
            if term or trunc:
                break
        foe = "blue" if seat == "red" else "red"
        rows.append(dict(
            seed=seed,
            outcome=info.get("outcome"),          # kill / died / mutual / timeout
            won=info.get("won"),
            t=round(info["t"], 1),                # when it ended [s]
            steps=steps,
            truncated=bool(trunc),                # ran out the clock
            own_health=round(info["own_health"], 4),
            opp_health=round(info["opp_health"], 4),
            # time I could shoot / time I sat where I could be shot
            wez_time=round(c.wez_time[seat], 2),
            wez_time_foe=round(c.wez_time[foe], 2),
            # time to the first successful aim.  None if it never aimed
            first_wez=(None if c.first_wez[seat] is None
                       else round(c.first_wez[seat], 1)),
            # longest unbroken track.  Below `track_lock`, damage is 0
            longest_track=round(longest, 2),
            min_range=round(c.min_range),
            out_of_bounds=bool(c.out_of_bounds(seat)),
        ))
    return rows


def summarise(rows: list[dict], seat: str = "red") -> dict:
    """Per-engagement rows -> one summary."""
    n = len(rows)
    kills = sum(1 for r in rows if r["won"])
    ts = [r["t"] for r in rows if r["won"]]
    firsts = [r["first_wez"] for r in rows if r["first_wez"] is not None]
    return dict(
        n=n, kills=kills, rate=round(kills / n, 4),
        t_kill=round(sum(ts) / len(ts), 1) if ts else None,
        outcomes={o: sum(1 for r in rows if r["outcome"] == o)
                  for o in sorted({r["outcome"] for r in rows if r["outcome"]})},
        own_health=round(sum(r["own_health"] for r in rows) / n, 4),
        opp_health=round(sum(r["opp_health"] for r in rows) / n, 4),
        wez_time=round(sum(r["wez_time"] for r in rows) / n, 2),
        wez_time_foe=round(sum(r["wez_time_foe"] for r in rows) / n, 2),
        # share that aimed at all, and how long that took when it happened
        engaged=round(len(firsts) / n, 3),
        first_wez=round(sum(firsts) / len(firsts), 1) if firsts else None,
        longest_track=round(sum(r["longest_track"] for r in rows) / n, 2),
        min_range=round(sum(r["min_range"] for r in rows) / n),
        out_of_bounds=round(sum(r["out_of_bounds"] for r in rows) / n, 3),
    )


def score(design: pathlib.Path, weights: pathlib.Path, proj,
          seed0: int, n: int) -> dict:
    """One design's score -- the rows and their summary."""
    act, note, mode = policies.load(design, weights)
    seats = (("red", "blue") if proj.DIR.name in P.SYMMETRIC_SEATS
             else ("red",))
    episodes = []
    for seat in seats:
        env = proj.ENV(action_mode=mode, seat=seat)
        # The same seeds in both seats: one engagement seen from each side, not
        # two different ones.
        rows = play(env, act, seed0, n // len(seats), seat=seat)
        for r in rows:
            r["seat"] = seat
        episodes += rows
    # Relative to the design: this file lands beside the weights, so an
    # absolute path is just the name of one laptop.
    try:
        shown = weights.resolve().relative_to(design.resolve()).as_posix()
    except ValueError:
        shown = weights.as_posix()
    per_seat = ({seat: summarise([r for r in episodes if r["seat"] == seat])
                 for seat in seats} if len(seats) > 1 else None)
    return dict(submission=design.name, policy=shown,
                policy_abs=str(weights.resolve()), note=note, band=seed0,
                seats=list(seats), summary=summarise(episodes),
                per_seat=per_seat, episodes=episodes)


def main(argv=None) -> int:
    name, rest = P.split_argv(argv)
    proj = P.load(name)

    ap = argparse.ArgumentParser(prog=f"tools.grade {name}", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default=None,
                    help="a folder with policy.py, wrappers.py and the "
                         "weights.  Default: the assignment's own `baseline*/`")
    ap.add_argument("--policy", default=None, help="override the weights file")
    ap.add_argument("--sweep", default=None,
                    help="score every <dir>/*/policy_net.* instead, using the "
                         "design's State.  For development, not marking")
    ap.add_argument("--n", type=int, default=STANDARD_N)
    ap.add_argument("--band", type=int, default=TEST_BAND,
                    help="first seed of the test band.  The default is "
                         "readable from this repository; real marking should "
                         "pass one that is not")
    ap.add_argument("--out", metavar="PATH",
                    help="where to write the JSON.  Default: score.json next "
                         "to the weights")
    args = ap.parse_args(rest)

    design = P.design_path(proj, args.design)
    try:
        if args.sweep:
            root = pathlib.Path(args.sweep)
            runs = sorted(w for name in policies.WEIGHTS
                          for w in root.glob(f"*/{name}"))
            if not runs:
                raise SystemExit(f"no {' or '.join(policies.WEIGHTS)} "
                                 f"under {args.sweep}")
            rows = []
            for w in runs:
                try:
                    r = score(design, w, proj, args.band, args.n)
                    r["submission"] = w.parent.name
                except policies.Mismatch as e:
                    r = dict(submission=w.parent.name, error=str(e))
                rows.append(r)
        else:
            weights = (pathlib.Path(args.policy) if args.policy
                       else policies.default_weights(design))
            rows = [score(design, weights, proj, args.band, args.n)]
    except policies.Mismatch as e:
        raise SystemExit(f"cannot grade: {e}")

    print(f"\n  {proj.TITLE} · band {args.band} + 0..{args.n - 1}, greedy, "
          f"n={args.n}\n")
    for r in sorted(rows, key=lambda r: -(r.get("summary") or {}).get("rate", -1.0)):
        if "error" in r:
            print(f"  {r['submission'][:44]:44s}  -- {r['error'][:70]}")
            continue
        m = r["summary"]
        seats = ("   " + "  ".join(f"{k} {v['kills']}/{v['n']}"
                                   for k, v in r["per_seat"].items())
                 if r.get("per_seat") else "")
        print(f"  {r['submission'][:44]:44s}  {m['kills']:>2}/{m['n']} = "
              f"{m['rate']:.2f}  t_kill {m['t_kill']}"
              f"  WEZ {m['wez_time']}s  exposed {m['wez_time_foe']}s{seats}")
    scored = [r for r in rows if "summary" in r]
    if not scored:
        return 1
    best = max(scored, key=lambda r: r["summary"]["rate"])
    if len(scored) > 1:
        print(f"\n  best: {best['submission']}  {best['summary']['rate']:.2f}")
    print()

    # Beside the weights by default.  `--sweep` measures many at once and sits
    # beside none of them, so there it writes only when `--out` says where.
    out = (pathlib.Path(args.out) if args.out else
           None if args.sweep else
           pathlib.Path(best["policy_abs"]).parent / "score.json")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        clean = [{k: v for k, v in r.items() if k != "policy_abs"} for r in rows]
        out.write_text(json.dumps(dict(band=args.band, n=args.n, rows=clean),
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
