# aircombat_gym.wvr — API

What the WVR environments take and hand back. Every environment in [README.md](README.md) has exactly this surface.

```python
obs, info = env.reset(seed=None)
obs, reward, terminated, truncated, info = env.step(action)
```

| returned     | type                       | what it is for                                                        |
| ------------ | -------------------------- | --------------------------------------------------------------------- |
| `obs`        | `np.ndarray`, `(39,)`, float32 | **what a policy sees.** The only input `Policy.act` gets            |
| `reward`     | `float`                    | **always 0.0.** You write the reward                                  |
| `terminated` | `bool`                     | the engagement ended in a result: `kill`, `died` or `mutual`          |
| `truncated`  | `bool`                     | the clock ran out: `timeout`                                          |
| `info`       | `dict`                     | **the material a reward is built from.** Training only — a policy never gets it |

Everything is written from the seat the learner flies (`red` unless the environment is built with `seat="blue"`): `own_` is you, `opp_` is the other aircraft.

## Constructor

```python
gym.make("AirCombat/FairFight-v0", action_mode="discrete", seed=None)
```

| argument      | default      | meaning                                         |
| ------------- | ------------ | ----------------------------------------------- |
| `action_mode` | `"discrete"` | `"discrete"` → `Discrete(9)`, `"continuous"` → `Box(2)` |
| `seed`        | `None`       | seeds the environment's own generator           |
| `seat`        | `"red"`      | which aircraft you fly                          |

## Action

**Type** — depends on `action_mode`:

- `Discrete(9)`: an `int` in 0–8. Anything outside raises `ValueError`
- `Box(2)`: an `np.ndarray` of shape `(2,)`, `float32`. Values outside [-1, 1] are clipped

One decision every 0.05 s (20 Hz). An action is a **change** to the autopilot's target heading and speed; a zero change keeps the current target.

**`Discrete(9)`** — `index = 3 × heading + speed`

|                   | speed −20 kt | speed 0 | speed +20 kt |
| ----------------- | :----------: | :-----: | :----------: |
| **heading −30°**  | 0            | 1       | 2            |
| **heading 0**     | 3            | 4 (no-op) | 5          |
| **heading +30°**  | 6            | 7       | 8            |

**`Box(2)`** — `[heading, speed]`, each in [-1, 1], scaled to ±30° and ±20 kt.

Heading + is a turn to the right. Altitude is not an action: both aircraft are held at 20,000 ft.

## Observation — `obs`, 39 channels

**Type** — `np.ndarray` of shape `(39,)`, `float32`. `env.observation_space` is `Box(-inf, inf, (39,), float32)`.

Raw SI values. Nothing is normalised, clipped or wrapped. Defined in [`obs.py`](obs.py); `O.unpack(obs)` returns it as a dict keyed by these names.

**Per aircraft** — `own_` at index 0–14, `opp_` at index 15–29, same order

| index (own / opp) | name                  | unit  | meaning                                          |
| ----------------- | --------------------- | ----- | ------------------------------------------------ |
| 0 / 15            | `x`                   | m     | position, east                                   |
| 1 / 16            | `y`                   | m     | position, north                                  |
| 2 / 17            | `h`                   | m     | altitude above sea level                         |
| 3 / 18            | `vx`                  | m/s   | velocity, east                                   |
| 4 / 19            | `vy`                  | m/s   | velocity, north                                  |
| 5 / 20            | `vz`                  | m/s   | velocity, **up**                                 |
| 6 / 21            | `nx`                  | g     | load factor, body forward                        |
| 7 / 22            | `ny`                  | g     | load factor, body right wing                     |
| 8 / 23            | `nz`                  | g     | load factor, body vertical; +1 in level flight   |
| 9 / 24            | `phi`                 | rad   | bank                                             |
| 10 / 25           | `theta`               | rad   | pitch                                            |
| 11 / 26           | `psi`                 | rad   | heading: 0 north, + clockwise                    |
| 12 / 27           | `p`                   | rad/s | roll rate                                        |
| 13 / 28           | `q`                   | rad/s | pitch rate                                       |
| 14 / 29           | `r`                   | rad/s | yaw rate                                         |

**Engagement**

| index | name                   | unit | meaning                                              |
| ----- | ---------------------- | ---- | ---------------------------------------------------- |
| 30    | `own_health`           | —    | 1.0 down to 0                                        |
| 31    | `opp_health`           | —    | 1.0 down to 0                                        |
| 32    | `own_track_time`       | s    | how long you have held a firing solution, unbroken   |
| 33    | `opp_track_time`       | s    | how long he has held one on you                      |
| 34    | `own_in_wez`           | 0/1  | you can shoot                                        |
| 35    | `opp_in_wez`           | 0/1  | he can shoot you                                     |
| 36    | `own_dist_to_boundary` | m    | to the arena edge (radius 50 km)                     |
| 37    | `opp_dist_to_boundary` | m    | the same, for him                                    |
| 38    | `t_remaining`          | s    | time left on the clock                               |

Swap the seats and every `own_`/`opp_` pair swaps — the same vector, seen from the other cockpit. Channels that are constant in a task (altitude, pitch, climb rate) stay in, so a policy trained on one environment loads into another.

## `info`

**Type** — `dict`, keyed by `str`; values are `float`, `bool` or `str` as listed below. A new dict every step.

The geometry already worked out — range, angles, closure — plus health and time. Defined in [`envs/base.py`](envs/base.py).

**Every step** (and from `reset`)

| key                    | type  | unit | meaning                                                                 |
| ---------------------- | ----- | ---- | ----------------------------------------------------------------------- |
| `range`                | float | m    | distance to him                                                         |
| `range_rate`           | float | m/s  | rate of change of `range`; **+ opening**                                |
| `ata`                  | float | rad  | 0–π. Your nose against the line of sight; 0 = he is dead ahead          |
| `ata_signed`           | float | rad  | −π–π. The same, horizontal and signed; + = he is to your right          |
| `ata_lead`             | float | rad  | 0–π. Your nose against the **aim point** — where the rounds meet him. The firing envelope is judged on this |
| `lead_signed`          | float | rad  | −π–π. Bearing of the aim point; + = right                               |
| `aa`                   | float | rad  | 0–π. His tail against the line of sight; 0 = you are on his six         |
| `in_wez`               | bool  | —    | you can shoot: `ata_lead` ≤ 30° and 150 m ≤ `range` ≤ 1,500 m           |
| `opp_in_wez`           | bool  | —    | he can shoot you                                                        |
| `track_time`           | float | s    | your unbroken firing solution. Damage starts at 1.0 s                   |
| `under_track`          | float | s    | his unbroken solution on you                                            |
| `damage_rate`          | float | 1/s  | reference only: the three-factor weapon model. The tasks apply a flat 0.33/s after the lock |
| `own_health`           | float | —    | 1.0 down to 0                                                           |
| `opp_health`           | float | —    | 1.0 down to 0                                                           |
| `own_speed`            | float | kt   | your true airspeed                                                      |
| `opp_speed`            | float | kt   | his                                                                     |
| `own_alt`              | float | ft   | your altitude                                                           |
| `alt_diff`             | float | ft   | his altitude minus yours                                                |
| `t`                    | float | s    | elapsed                                                                 |
| `t_remaining`          | float | s    | left                                                                    |
| `dist_to_boundary`     | float | m    | to the arena edge                                                       |
| `opp_dist_to_boundary` | float | m    | the same, for him                                                       |

**Added on the last step**

These keys are **not in the dict** before the last step — absent, not `None`. `info["won"]` mid-engagement raises `KeyError`; read them with `info.get("won")`, which gives `None` until then.

| key        | type  | meaning                                                                           |
| ---------- | ----- | --------------------------------------------------------------------------------- |
| `outcome`  | str   | `kill` · `died` · `timeout`, and in the two duels (`AdvantagedFight`, `FairFight`) `mutual`: both down, or a kill that leaves you on 0.02 health or less |
| `won`      | bool  | `True` for `kill` only                                                            |
| `wez_time` | float | total time you could shoot in this engagement [s]                                 |