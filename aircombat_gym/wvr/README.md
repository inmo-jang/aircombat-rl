# aircombat_gym.wvr — within visual range

Gun fights between two F-16s: the Gym environments, the 39-channel observation, the action grid, the gun and the scripted opponents. The aircraft underneath — flight model, autopilot, TacView output — is shared and described in [`aircombat_gym/README.md`](../README.md).

## Available `Gym` environments

| id                                                                                 | the other aircraft                                                      | how it starts                                                                              | timeout |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------- |
| [`AirCombat/Circular-v0`](../../templates/project_01_circular/README.md)          | orbits at a fixed rate. Never evades, never aims                        | you 1.5–4 km out, nose within 90° of it                                                  | 120 s   |
| [`AirCombat/Evader-v0`](../../templates/project_02_evader/README.md)              | cruises straight, then breaks hard in a random direction when you close | same                                                                                       | 240 s   |
| [`AirCombat/AdvantagedFight-v0`](../../templates/project_03_advantaged/README.md) | lead pursuit with closure control — the strongest one here             | 4–10 km apart, your nose within 90° of him and his free, so the opening is usually yours | 120 s   |
| [`AirCombat/FairFight-v0`](../../templates/project_04_fair/README.md)             | the same one                                                            | abeam, 10 km, opposite headings — dead even                                               | 120 s   |

Whichever you pick:

- **observation space** — the same 39 raw channels every time
- **action space** — two axes, heading and speed. `Discrete(9)` by default, three values each; `Box(2)` in [-1, 1] with `action_mode="continuous"`
- **both aircraft held at 20,000 ft** — the third axis is closed, so nothing you send can move it, and the autopilot's altitude hold flies it for you

### How to Use

```python
import gymnasium as gym
import aircombat_gym.wvr.envs                 # registers the ids

env = gym.make("AirCombat/Circular-v0")       # obs 39, action Discrete(9)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

**The environment does not compute a reward** — `step` always returns 0.0, and the reward is yours to write. What `reset` and `step` hand back — the 39-channel `obs` a policy sees, and the `info` dict a reward is built from — is listed in **[API.md](API.md)**.

## Try by keyboard

```bash
python -m aircombat_gym.wvr.play                                 # circular
python -m aircombat_gym.wvr.play --env evader
python -m aircombat_gym.wvr.play --env advantaged --continuous
python -m aircombat_gym.wvr.play --env fair --tacview
```

Four arrow keys, and each press becomes an element of `env.action_space` and goes through `env.step()` — same weapon, same opponent, same `info`, no shortcut around the environment. **Fly a task before you write a reward for it**, or the reward is a guess about a game you have not played.

| key           | action                                                                                       |
| ------------- | -------------------------------------------------------------------------------------------- |
| `←` `→` | heading −30° / +30°                                                                       |
| `↑` `↓` | speed +20 kt / −20 kt                                                                       |
| nothing held  | the no-op action, which *freezes* the target rather than re-deriving it                    |
| `m`         | swap `Discrete(9)` for `Box(2)` mid-engagement — the same fight, the other action space |

**Action spaces** — what one action is

- `Discrete(9)` (default) — an index 0–8: heading {−30°, 0, +30°} × speed {−20, 0, +20 kt}; `4` is the no-op
- `Box(2)` (`--continuous`) — `[heading, speed]`, each in [-1, 1], scaled to ±30° and ±20 kt

**Other keys** — the same as in the [hand-flying tool](../README.md#keys)

- `BACKSPACE` — new engagement
- `p` — pause
- `+` `−` — zoom
- `esc` — quit

**Environments** — `--env`

| `--env`              | runs                             | in short                                                       |
| ---------------------- | -------------------------------- | -------------------------------------------------------------- |
| `circular` (default) | `AirCombat/Circular-v0`        | a target in a steady turn                                      |
| `evader`             | `AirCombat/Evader-v0`          | a target that breaks hard when you close                       |
| `advantaged`         | `AirCombat/AdvantagedFight-v0` | a duel against a bot that shoots back; you start with the edge |
| `fair`               | `AirCombat/FairFight-v0`       | the same duel from an even start                               |

**Etc**

- `--tacview` — streams the engagement live to TacView Advanced (Windows only): in TacView, *Record → Real-time Telemetry*, and enter the address the command prints. Leave it off on macOS and Linux.
