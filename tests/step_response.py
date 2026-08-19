"""Step-response benchmark for the autopilot (gate B1).

Three commands, each from trimmed level flight, each timed to settle:

    altitude   -> 30,000 ft   (the top of the band)
    speed      -> 650 kt      (the top of the band)
    heading    reverse 180 deg

Three columns, because settling time on its own lies.  A controller can
"arrive" fast by overshooting hugely, and it can buy altitude by spending speed.
So `overshoot` (counted only after the target is first reached) and
`side effect` (what the other axes gave up) are there to catch that.

Run it after touching anything in `core/control/autopilot.py`.  The gains there
were set from these numbers -- the tables in that module's comments are this
tool's output at 320/450/600 kt.

It lives here rather than in the package because it measures the environment
instead of being part of it: no student needs it, and `pip install
aircombat-gym` should not ship a controller benchmark.  It is not a pytest test
either -- there is no pass/fail line to draw, only numbers to compare against
the last time -- so pytest does not collect it and you run it by hand.

    python tests/step_response.py
    python tests/step_response.py --speed 320
"""
from __future__ import annotations

import argparse
import math

from aircombat_gym.core.aircraft import Aircraft
from aircombat_gym.core.control.autopilot import Autopilot, wrap_pi
from aircombat_gym.core.envelope import (ALT_MAX_FT, ALT_MIN_FT, V_MAX_KT,
                                         V_MIN_KT)


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _run(alt, v, *, d_alt=0.0, d_v=0.0, d_psi=0.0, t_max=180.0,
         gains=None, tol_alt=200.0, tol_v=10.0, tol_psi=2.0):
    """One step command.  Returns (t_settle, peak overshoot, worst side effect)."""
    ac = Aircraft(h0_ft=alt, autopilot=Autopilot(h0_ft=alt, gains=gains))
    ac.reset(v_kt=v)
    for _ in range(60):                      # let the trim settle
        ac.hold()
    s = ac.state
    ac.alt_cmd_ft = _clip(alt + d_alt, ALT_MIN_FT, ALT_MAX_FT)
    ac.v_cmd_kt = _clip(v + d_v, V_MIN_KT, V_MAX_KT)
    ac.psi_cmd = wrap_pi(s.psi + math.radians(d_psi))

    target = (ac.alt_cmd_ft, ac.v_cmd_kt, ac.psi_cmd)
    t_settle, over, side, arrived = None, 0.0, 0.0, False
    for i in range(1, int(t_max * 20) + 1):
        s = ac.hold()
        if not s.flyable:
            return None, float("nan"), float("nan")
        if d_alt:
            e, done = s.h_ft - target[0], abs(s.h_ft - target[0]) <= tol_alt
            side = max(side, abs(s.v_kt - v))              # speed lost climbing
        elif d_v:
            e, done = s.v_kt - target[1], abs(s.v_kt - target[1]) <= tol_v
            side = max(side, abs(s.h_ft - alt))            # altitude wandered
        else:
            e = -math.degrees(wrap_pi(target[2] - s.psi))
            done = abs(e) <= tol_psi
            side = max(side, abs(s.h_ft - alt))
        # Overshoot only counts once the target has been reached: a 180 deg
        # reversal starts 180 deg away, and calling that "overshoot" made the
        # column read 180.0 for every controller.
        arrived = arrived or done
        if arrived:
            over = max(over, e if (d_alt or d_v or d_psi) > 0 else -e)
        if done and t_settle is None:
            t_settle = i / 20.0
    return t_settle, over, side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speed", type=float, default=450.0, help="starting KTAS")
    ap.add_argument("--alt", type=float, default=20000.0, help="starting altitude")
    args = ap.parse_args(argv)

    v, alt = args.speed, args.alt

    tests = (
        ("altitude  -> 30,000 ft", dict(d_alt=ALT_MAX_FT - alt), "ft", "kt lost"),
        ("speed     -> 650 kt", dict(d_v=V_MAX_KT - v), "kt", "ft of alt"),
        ("heading   reverse 180", dict(d_psi=180.0), "deg", "ft of alt"),
    )
    print(f"start: {v:.0f} kt at {alt:,.0f} ft\n")
    print(f"{'step':<24}{'settle':>9}{'overshoot':>12}{'side effect':>16}")
    print("-" * 61)
    for label, kw, unit, side_unit in tests:
        t, over, side = _run(alt, v, **kw)
        ts = f"{t:7.1f} s" if t else "     --  "
        print(f"{label:<24}{ts:>9}{over:9.1f} {unit:<3}{side:9.0f} {side_unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
