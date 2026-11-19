"""Fly `red` against `blue`, both already trained, on the starts in `config.yaml`.

    python match.py                              # the score
    python match.py --watch                      # on screen, until ESC
    python match.py --watch --n 2                # on screen, 2 seeds x both arrangements
    python match.py --watch --speed 4            # 4x; + and - double and halve it
    python match.py --watch --tacview            # live in TacView too (Windows)
    python match.py --red players/a --blue ace   # override the config's teams
    python match.py --out                        # also results/<red>_vs_<blue>_<yymmdd>_<hhmm>.json

Both modes fly the same engagements in the same order.  With `swap_seats`,
every seed is flown twice -- the starts as configured, then the two starts
traded -- so a start that favours one side shows up as a gap between the two
rows rather than as a win for either team.  Each team keeps its colour
throughout: `red` is red, on screen and in TacView, whichever start it has.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json

import arena

OUTCOMES = ("kill", "died", "mutual", "timeout")
LABEL = {"kill": "red wins", "died": "blue wins", "mutual": "mutual",
         "timeout": "timeout"}
ARRANGEMENT = {False: "as configured", True: "swapped"}


def _team(cfg, side, override):
    if override:
        cfg = dict(cfg, **{side: override})
    return arena.load_team(arena.team_path(cfg, side))


def schedule(cfg, seed0: int):
    """(seed, swapped) for every engagement, in flying order, without end."""
    for i in itertools.count():
        if cfg["swap_seats"]:
            yield seed0 + i // 2, bool(i % 2)
        else:
            yield seed0 + i, False


class Tally:
    """Outcome counts per arrangement, and one record per engagement."""

    def __init__(self):
        self.rows: dict = {}
        self.records: list = []

    def add(self, seed, swapped, info) -> None:
        label = ARRANGEMENT[swapped]
        self.rows.setdefault(label, dict.fromkeys(OUTCOMES, 0))[info["outcome"]] += 1
        self.records.append(dict(
            seed=seed, starts=label, result=LABEL[info["outcome"]],
            left_altitude_band=info.get("disqualified"),
            t=round(info["t"], 1), red_health=round(info["own_health"], 3),
            blue_health=round(info["opp_health"], 3)))

    def ordered(self) -> dict:
        return {a: self.rows[a] for a in ARRANGEMENT.values() if a in self.rows}


def arena_env(cfg, red, blue):
    """The env a match is flown in: `red` in the red seat, `blue` the opponent."""
    return arena.ArenaEnv(cfg["start"], blue, action_mode=red.mode,
                          altitude_band_ft=cfg["altitude_band_ft"],
                          time_limit_s=cfg["time_limit_s"], decide_on_health=True)


def play(cfg, red, blue, plan, tally: Tally) -> None:
    """Every engagement in `plan`, without a window."""
    env = arena_env(cfg, red, blue)
    for seed, swapped in plan:
        obs, _ = env.reset(seed=seed, options={"swapped": swapped})
        while True:
            obs, _, term, trunc, info = env.step(red.act(obs))
            if term or trunc:
                tally.add(seed, swapped, info)
                break


def report(red, blue, rows: dict, records: list = ()) -> None:
    """Outcome counts per arrangement -> the table, and who left the band."""
    if not rows:
        print("\n  no engagement ran to the end")
        return
    print(f"\n  {'starts':16s}" + "".join(f"{LABEL[o]:>11s}" for o in OUTCOMES))
    total = dict.fromkeys(OUTCOMES, 0)
    for label, c in rows.items():
        print(f"  {label:16s}" + "".join(f"{c[o]:>11d}" for o in OUTCOMES))
        for o in OUTCOMES:
            total[o] += c[o]
    if len(rows) > 1:
        print(f"  {'total':16s}" + "".join(f"{total[o]:>11d}" for o in OUTCOMES))
    played = sum(total.values())
    print(f"\n  {red.name} {total['kill']} - {total['died']} {blue.name}"
          f"   ({played} engagements)")
    left = [r["left_altitude_band"] for r in records if r.get("left_altitude_band")]
    if left:
        print(f"  left the altitude band: {red.name} {left.count('red')}, "
              f"{blue.name} {left.count('blue')}, both {left.count('both')}")


def save(cfg, red, blue, tally: Tally, mode: str) -> None:
    """The match to `results/<red>_vs_<blue>_<yymmdd>_<hhmm>.json`.

    `results/` sits next to `config.yaml`.  A file of the same name -- two
    matches in one minute -- is never overwritten; the second gets `_2`.
    """
    if not tally.records:
        print("  nothing to save")
        return
    now = datetime.datetime.now()
    folder = cfg["_dir"] / "results"
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{red.name}_vs_{blue.name}_{now:%y%m%d_%H%M}"
    path, k = folder / f"{base}.json", 2
    while path.exists():
        path, k = folder / f"{base}_{k}.json", k + 1
    rows = tally.ordered()
    total = {o: sum(c[o] for c in rows.values()) for o in OUTCOMES}
    data = dict(
        red=dict(name=red.name, policy=red.note),
        blue=dict(name=blue.name, policy=blue.note),
        when=now.isoformat(timespec="seconds"), mode=mode,
        start=dict(cfg["start"]), altitude_band_ft=cfg["altitude_band_ft"],
        score={LABEL[o]: total[o] for o in OUTCOMES} | {"engagements": len(tally.records)},
        per_arrangement={a: {LABEL[o]: c[o] for o in OUTCOMES} for a, c in rows.items()},
        engagements=tally.records,
    )
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="default: config.yaml next to this file")
    ap.add_argument("--red", default=None, help="override config `red`")
    ap.add_argument("--blue", default=None, help="override config `blue`")
    ap.add_argument("--n", type=int, default=None,
                    help="seeds, each flown in every arrangement.  Default: "
                         "config `engagements`; with --watch, until ESC")
    ap.add_argument("--seed0", type=int, default=1_000_000,
                    help="first seed; seed i is seed0 + i")
    ap.add_argument("--watch", action="store_true", help="on screen")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="--watch only: replay speed, 1 = real time, 0 = as "
                         "fast as it goes.  + and - double and halve it")
    ap.add_argument("--tacview", action="store_true",
                    help="--watch only: live TacView (Windows)")
    ap.add_argument("--out", action="store_true",
                    help="also save the result to results/<red>_vs_<blue>_"
                         "<yymmdd>_<hhmm>.json, next to config.yaml")
    args = ap.parse_args(argv)

    cfg = arena.load_config(args.config)
    red = _team(cfg, "red", args.red)
    blue = _team(cfg, "blue", args.blue)
    if red.scripted:
        raise SystemExit("`red` must be a policy folder; a scripted pilot can "
                         "only fly `blue`")
    print(f"  red   {red.name:28s} {red.note}")
    print(f"  blue  {blue.name:28s} {blue.note}")

    per_seed = 2 if cfg["swap_seats"] else 1
    tally = Tally()
    if args.watch:
        import viewer
        sinks = []
        if args.tacview:
            from aircombat_gym.core.tacview import AcmiRealtime
            live = AcmiRealtime()
            sinks.append(live)
            print(f"  TacView real-time telemetry: {live.address}")
            print("    TacView: Record -> Real-time Telemetry -> that address")
        plan = schedule(cfg, args.seed0)
        if args.n:
            plan = itertools.islice(plan, args.n * per_seed)
        env = arena_env(cfg, red, blue)
        viewer.watch(env, red, blue, plan, speed=args.speed, sinks=sinks,
                     on_result=tally.add)
        for s in sinks:
            s.close()
    else:
        n = args.n or int(cfg["engagements"])
        play(cfg, red, blue, itertools.islice(schedule(cfg, args.seed0), n * per_seed),
             tally)

    report(red, blue, tally.ordered(), tally.records)
    if args.out:
        save(cfg, red, blue, tally, "watch" if args.watch else "score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
