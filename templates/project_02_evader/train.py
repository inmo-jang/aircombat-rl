"""Training loop.

    build learner -> learn a chunk -> score on `val` -> keep the best snapshot

Writes `results/s<seed><tag>/policy_net.zip` and `result.json`.
Final score comes from `tools.grade`, not from here.
Algorithm, network, hyperparameters: `<design>/policy.py`.
"""
from __future__ import annotations

# =============================================================================
# Validation settings
# =============================================================================

#: steps between `val` scorings
VAL_EVERY_STEPS = 25_000

# A seed fixes one engagement, so `<seed0> + 0..n-1` is a fixed set of them.
# Held apart from the seeds training draws on.
VAL_SEED0, VAL_N = 900_000, 30    # 900,000~900,029.  picks which snapshot to keep

# =============================================================================

import argparse                                                      # noqa: E402
import json                                                          # noqa: E402
import sys                                                           # noqa: E402
import time                                                          # noqa: E402
from pathlib import Path                                             # noqa: E402

import torch                                                         # noqa: E402

HERE = Path(__file__).resolve().parent

# The design folder goes on the path before its modules are imported, so
# `--design` has to be read here rather than in `main`.
DESIGN = "baseline"
for _i, _a in enumerate(sys.argv):
    if _a == "--design" and _i + 1 < len(sys.argv):
        DESIGN = sys.argv[_i + 1]
    elif _a.startswith("--design="):
        DESIGN = _a.split("=", 1)[1]

sys.path.insert(0, str(HERE / DESIGN))

from policy import Policy                                            # noqa: E402
from wrappers import make_env                                        # noqa: E402


def evaluate(act, env, seed0: int, n: int) -> dict:
    """Run `n` engagements from seed `seed0`; count kills and mean time-to-kill.

        act(obs)  one raw observation -> one action
        env       raw-observation env, i.e. `make_env(shaped=False)`
    """
    kills, ts = 0, []
    for i in range(n):
        obs, _ = env.reset(seed=seed0 + i)
        while True:
            obs, _, term, trunc, info = env.step(act(obs))
            if term or trunc:
                if info.get("won"):
                    kills += 1
                    ts.append(info["t"])
                break
    return dict(kills=kills, n=n, rate=kills / n,
                t_kill=(sum(ts) / len(ts) if ts else float("nan")))


def train(args) -> dict:
    # --- eval env ------------------------------------------------------------
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    eval_env = make_env(shaped=False)

    # --- output directory ----------------------------------------------------
    out = Path(args.out) / args.design / f"s{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)

    # --- learner -------------------------------------------------------------
    model = Policy.make_learner(make_env, args.seed, device)
    training_policy = Policy(model=model)      # same object as `model`

    print(f"=== seed {args.seed} / {training_policy} / {args.steps:,} steps / {device} ===",
          flush=True)

    # --- training loop -------------------------------------------------------
    best_score, t0 = (-1, 0.0), time.time()
    rounds = max(1, args.steps // args.val_every)
    for i in range(1, rounds + 1):
        model.learn(total_timesteps=args.val_every, reset_num_timesteps=False,
                    progress_bar=False)
        m = evaluate(training_policy.act, eval_env, VAL_SEED0, VAL_N)

        # Save only on improvement, so the file holds the best snapshot rather
        # than the last one.  Kills first, faster kill breaks the tie.
        current_score = (m["kills"],
                         -(m["t_kill"] if m["t_kill"] == m["t_kill"] else 1e9))
        if current_score > best_score:
            best_score = current_score
            model.save(out / "policy_net")

        print(f"  {i * args.val_every:>9,} steps   val {m['kills']:>2}/{VAL_N}   "
              f"{(time.time() - t0) / 60:.0f}m", flush=True)

    # --- record --------------------------------------------------------------
    result = dict(seed=args.seed, policy=str(training_policy),
                  steps=args.steps, val_best=best_score[0], val_n=VAL_N,
                  minutes=round((time.time() - t0) / 60, 1))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nval best {best_score[0]}/{VAL_N}  in {result['minutes']:.0f} min"
          f"  ->  {out / 'policy_net.zip'}", flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", default=DESIGN,
                    help="folder with policy.py and wrappers.py.  Read before "
                         "the imports above; listed here so --help shows it")
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
