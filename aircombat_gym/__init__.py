"""aircombat_gym -- air combat environments on JSBSim's F-16.

One scenario so far: `wvr`, a 1v1 gun fight.  The split below is what lets the
next one cost less than the first.

    core/       the aircraft and how it is flown.  JSBSim backend, guidance,
                the measured flight envelope, TacView output.  No task knows
                about another and none of this knows about any of them.
    wvr/        within visual range: the gun task -- observation spec and
                weapon model.  A `bvr/` sits beside it when missiles arrive.
    baselines   scripted opponents, which are part of the environment rather
                than part of anyone's training loop
    tools/      things you point at the environment: hand flying, step response

Nothing here imports a learning library.  Training and grading live outside the
package and consume it, which is the same boundary D10 rests on: the environment
does not compute reward, the student does.
"""
from .core.envelope import (ALT_MAX_FT, ALT_MIN_FT, H0_FT, THROTTLE_CAP,
                            V_MAX_KT, V_MIN_KT, max_bank_deg)
from .wvr.spaces import OBS_SPEC_VERSION, action_n, obs_dim

__all__ = ["H0_FT", "ALT_MIN_FT", "ALT_MAX_FT", "THROTTLE_CAP",
           "V_MIN_KT", "V_MAX_KT", "max_bank_deg",
           "OBS_SPEC_VERSION", "obs_dim", "action_n"]
