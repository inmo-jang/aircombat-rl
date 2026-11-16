"""Training loop, against the opponent and on the starts `config.yaml` names.

    build learner -> learn a chunk -> score on `val` -> keep the best snapshot

    python train.py                        # trains config `red` against config `blue`
    python train.py --design my_folder     # trains another folder against `blue`

Writes `results/<design>/s<seed><tag>/policy_net.zip` and `result.json`.
Copy the one you want into your folder by hand; `match.py` reads it there.

The same loop as the other assignments' `train.py`.  What differs is the
environment: not `TASK` from your `wrappers.py`, but the arena from
`config.yaml`.  Your `State`, `RewardFunction` and `Shaped` are used as they
are; `TASK` and `make_env` are not.
"""
from __future__ import annotations

# =============================================================================
# Validation settings
# =============================================================================

#: steps between `val` scorings
VAL_EVERY_STEPS = 25_000

# A seed fixes one engagement, so `<seed0> + 0..n-1` is a fixed set of them.
# With `swap_seats` each is flown from both starts, so `VAL_N` is split in two.
VAL_SEED0, VAL_N = 900_000, 30

# =============================================================================

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import pathlib                                                       # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402

import torch                                                         # noqa: E402

import arena                                                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def _arg(name: str):
    """`--name value` or `--name=value` from argv, before argparse runs."""
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


# The design folder goes on the path before its modules are imported, so
# `--config` and `--design` are read here rather than in `main`.
CFG = arena.load_config(_arg("--config"))
_given = _arg("--design")
DESIGN = arena.team_path(dict(CFG, red=_given) if _given else CFG, "red")
if isinstance(DESIGN, str):
    raise SystemExit(f"`red` is the scripted pilot `{DESIGN}`: there is "
                     f"nothing to train.  Name a folder, or pass --design")
if not (DESIGN / "policy.py").exists():
    raise SystemExit(f"{DESIGN}/policy.py not found")

sys.path.insert(0, str(DESIGN))

from policy import Policy                                            # noqa: E402

try:
    from wrappers import ACTION_MODE, RewardFunction, Shaped, State  # noqa: E402
except ImportError as e:
    raise SystemExit(
        f"{DESIGN.name}/wrappers.py cannot be trained: {e}.\n"
        f"train.py needs ACTION_MODE, State, RewardFunction and Shaped, as the "
        f"template has.  A shared copy (tools.share) can only fly, not train.")
if not hasattr(Policy, "make_learner"):
    raise SystemExit(f"{DESIGN.name}/policy.py has no Policy.make_learner, so "
                     f"there is nothing to train.  A shared copy (tools.share) "
                     f"can only fly.")


def evaluate(act, env, seed0: int, n: int, swap: bool) -> dict:
    """`n` engagements from seed `seed0`; kills and mean time-to-kill.

    With `swap`, n/2 seeds flown from each start -- the same seeds both ways.
    """
    seats = arena.SIDES if swap else ("red",)
    kills, ts, played = 0, [], 0
    for seat in seats:
        for i in range(n // len(seats)):
            obs, _ = env.reset(seed=seed0 + i, options={"seat": seat})
            while True:
                obs, _, term, trunc, info = env.step(act(obs))
                if term or trunc:
                    played += 1
                    if info.get("won"):
                        kills += 1
                        ts.append(info["t"])
                    break
    return dict(kills=kills, n=played, rate=kills / max(played, 1),
                t_kill=(sum(ts) / len(ts) if ts else float("nan")))


def train(args) -> dict:
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    blue_cfg = dict(CFG, blue=args.blue) if args.blue else CFG
    blue = arena.load_team(arena.team_path(blue_cfg, "blue"))
    swap = bool(CFG["swap_seats"])

    def make_env(seed=None, shaped=True):
        env = arena.ArenaEnv(CFG["start"], blue, swap_seats=swap,
                             altitude_band_ft=CFG["altitude_band_ft"],
                             action_mode=ACTION_MODE, seed=seed)
        return Shaped(env, State(), RewardFunction()) if shaped else env

    eval_env = make_env(shaped=False)

    out = pathlib.Path(args.out) / DESIGN.name / f"s{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    model = Policy.make_learner(make_env, args.seed, device)
    training_policy = Policy(model=model)      # same object as `model`

    print(f"=== {DESIGN.name} vs {blue.name} ({blue.note}) ===")
    print(f"=== seed {args.seed} / {training_policy} / {args.steps:,} steps / "
          f"{device} ===", flush=True)

    best_score, t0 = (-1, 0.0), time.time()
    rounds = max(1, args.steps // args.val_every)
    for i in range(1, rounds + 1):
        model.learn(total_timesteps=args.val_every, reset_num_timesteps=False,
                    progress_bar=False)
        m = evaluate(training_policy.act, eval_env, VAL_SEED0, VAL_N, swap)

        # Save only on improvement, so the file holds the best snapshot rather
        # than the last one.  Kills first, faster kill breaks the tie.
        current_score = (m["kills"],
                         -(m["t_kill"] if m["t_kill"] == m["t_kill"] else 1e9))
        if current_score > best_score:
            best_score = current_score
            model.save(out / "policy_net")

        print(f"  {i * args.val_every:>9,} steps   val {m['kills']:>2}/{m['n']}   "
              f"{(time.time() - t0) / 60:.0f}m", flush=True)

    result = dict(design=DESIGN.name, opponent=blue.name, seed=args.seed,
                  policy=str(training_policy), steps=rounds * args.val_every,
                  val_best=best_score[0], val_n=m["n"],
                  minutes=round((time.time() - t0) / 60, 1))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nval best {best_score[0]}/{m['n']}  in {result['minutes']:.0f} min"
          f"  ->  {out / 'policy_net.zip'}", flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="default: config.yaml next to this file")
    ap.add_argument("--design", default=None,
                    help="the folder to train.  Default: config `red`")
    ap.add_argument("--blue", default=None,
                    help="the opponent.  Default: config `blue`")
    ap.add_argument("--seed", type=int, default=0, help="random stream of the run")
    ap.add_argument("--steps", type=int, default=Policy.TOTAL_STEPS,
                    help="default: `Policy.TOTAL_STEPS`")
    ap.add_argument("--val-every", dest="val_every", type=int,
                    default=VAL_EVERY_STEPS, help="steps between val scorings")
    ap.add_argument("--tag", default="", help="suffix for the result directory")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--cuda", action="store_true")
    train(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
