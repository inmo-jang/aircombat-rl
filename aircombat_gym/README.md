# aircombat_gym

Air combat on [JSBSim](https://github.com/JSBSim-Team/jsbsim)'s F-16 6-DOF
model — the real flight control system, not a point mass. The aircraft is
shared; each kind of engagement lives in its own subpackage.

## Install

```bash
pip install -e .
```

## Environments

| | |
| --- | --- |
| **WVR** — within visual range, guns | **[wvr/README.md](wvr/README.md)**: the Gym environments, how to use them, and flying one yourself through the action space |

## Try by keyboard

```bash
python -m aircombat_gym.tools.manual_operation --tacview                   # solo
python -m aircombat_gym.tools.manual_operation --tacview --enemy circler   # with a target
```

![Hand-flying: TacView on the left, the pygame panel on the right](docs/demo.gif)

> **On macOS or Ubuntu, drop `--tacview`** from anything below — TacView is Windows-only. Everything you need is in the pygame panel; you only lose the 3D view. `--acmi flight.acmi` records a file on any OS and opens on Windows later.

`--enemy` puts up one of the WVR scripted opponents and scores the gun exactly as
the [WVR environments](wvr/README.md) do.

## Two control modes

|             | **AUTOPILOT** (default)                    | **STICK**                     |
| ----------- | ------------------------------------------------ | ----------------------------------- |
| you command | target heading, speed, altitude                  | aileron, elevator, rudder, throttle |
| who flies   | the controller works out bank, g, thrust         | you                                 |
| why         | **this is the interface your policy uses** | to see what happens underneath      |

## Keys

Letters mirror the numpad 3×3, so it works one-handed and works on a laptop.

```
   Q W E          7 8 9          rudder / pitch / rudder
   A S D    ==    4 5 6          left / centre / right
   Z X C          1 2 3          throttle+ / pitch / throttle-
```

| axis       | keys          | **AUTOPILOT**      | **STICK**                                 |
| ---------- | ------------- | ------------------------ | ----------------------------------------------- |
| left/right | `A` `D`   | target heading ∓        | roll left / right                               |
| up/down    | `W` `X`   | target altitude ±       | **nose down / up** (stick forward / back) |
| fore/aft   | `Z` `C`   | target speed ±          | throttle up / down                              |
| rudder     | `Q` `E`   | —                       | rudder left / right                             |
| centre     | `S` / space | freeze all three targets | centre the stick                                |
| `shift`  |               | bigger step              | full deflection                                 |

Arrow keys, `Home`/`PgUp`/`End`/`PgDn`/`Ins`/`Del` and the numpad all work too,
NumLock on or off.

Also: `tab` layer · `m` CONTINUOUS↔DISCRETE (AUTOPILOT only) · `p` pause ·
`BACKSPACE` reset (new random engagement) · `esc` quit · `+` `−` zoom

## Layout

| | |
| --- | --- |
| `core/` | the aircraft: JSBSim F-16, autopilot, flight envelope, TacView output. No task and no weapon model, so a BVR package can reuse it unchanged |
| `tools/` | `manual_operation` (above) and the viewer `tools.watch` uses to replay a policy |
| `wvr/` | within-visual-range gun fights — [wvr/README.md](wvr/README.md) |
