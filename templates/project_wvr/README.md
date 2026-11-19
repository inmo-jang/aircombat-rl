# Project - WVR

Fly any two policies against each other.

```
project_wvr/
├── config.yaml              who flies, and where they start        <- you edit this
├── train.py                 trains `red` against `blue`
├── match.py                 flies `red` against `blue`
├── arena.py                 used by train.py and match.py
├── viewer.py                the match on screen
├── players/                 the teams, one folder each
│   ├── trainee/             an untrained team to start from
│   ├── policy_01_share/     practice opponents: they fly, but cannot be trained
│   └── policy_02_share/
└── results/                 written by train.py and match.py --out
```

## Rules

- Two F-16s at 20,000 ft, each with a gun. Keep the other aircraft in your gun sight for about 4 s without a break and it goes down.
- Shoot the other down and you win. Both down at once is `mutual`.
- Match only: when `time_limit_s` (120 s) runs out, more health wins; a near tie is a `timeout`.
- Leave 20,000 ft ± `altitude_band_ft` (1,000 ft) and you lose.

## `config.yaml`

Set up `config.yaml` before you fly or train: which two teams fly, where each starts, and the altitude rule.

| key                           | meaning                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `red`, `blue`             | a team folder as a path from `config.yaml` (`players/trainee`), or a scripted pilot: `ace`, `pursuit`, `lead`, `circler`, `evader` |
| `start.red`, `start.blue` | `x`, `y` [m] from the arena centre (x east, y north), `heading` [deg, 0 north, clockwise], `speed_kt`                                    |
| `start.randomize`           | `position_m`, `heading_deg`: per aircraft, up to ±this. `speed_frac`: both speeds by one factor, within ±this. 0 turns one off           |
| `altitude_band_ft`          | leave 20,000 ft ± this and you lose. 0 turns the rule off                                                                                       |
| `time_limit_s`              | `match.py`: the clock [s]; then more health wins                                                                                                |
| `swap_seats`                | `true`: every seed is also flown with the starts swapped                                                                                       |
| `engagements`               | `match.py`: the number of seeds                                                                                                                |

The arena is 50 km in radius. Keep the two starts on the same `y`: positions are projected onto the globe, so a north/south pair is not a mirror image.

## Fly a match

```bash
# at templates/project_wvr
python match.py                              # the score
python match.py --out --watch --tacview --red players/trainee --blue players/policy_02_share
                                             # on screen and live in TacView, these two teams, result saved
```

```
  starts             red wins  blue wins     mutual    timeout
  as configured             5          0          0          0
  swapped                   4          1          0          0
  total                     9          1          0          0
```

Each seed is flown twice: with the starts as in `config.yaml`, then with them swapped. Each team keeps its colour either way.

## Train

```yaml
# config.yaml
red:  players/trainee                 # trained
blue: players/policy_01_share         # the opponent
```

```bash
# at templates/project_wvr
python train.py
```

Each run starts from scratch and writes `results/trainee/s<seed>/policy_net.zip`; copy it into `players/trainee/` to fly it.

## Share your policy

A policy folder holds your reward and training code. To let a classmate fly against you without reading them, share a copy  :

```bash
# from the repository root
python -m tools.share templates/project_wvr/players/trainee --out templates/project_wvr/players/trainee_share
```

The copy keeps only your `State` and the network weights, and is checked to fly exactly like the original.
