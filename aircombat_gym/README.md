# aircombat_gym

An F-16 air combat environment on [JSBSim](https://github.com/JSBSim-Team/jsbsim)'s
6-DOF model — the real flight control system, running at 120 Hz, not a point mass.

**Fly it by hand before you train anything on it.** The point is to feel what your
policy will be commanding, and what the aircraft refuses to do.

> The RL surface (`reset` / `step`) is not built yet. This is the flight model,
> the control interface and the gun.

## Install and run

```bash
pip install -e .

python -m aircombat_gym.tools.manual_operation --tacview                   # solo
python -m aircombat_gym.tools.manual_operation --tacview --enemy circler   # with a target
```

**On macOS or Ubuntu, drop `--tacview`** — TacView is Windows-only. Everything you
need to fly is in the pygame panel; you only lose the 3D view. To watch in 3D later,
record with `--acmi flight.acmi` (works on any OS) and open the file on Windows.

## Two control layers — `tab` switches

| | **GUIDANCE** (default) | **STICK** |
|---|---|---|
| you command | target heading, speed, altitude | aileron, elevator, rudder, throttle |
| who flies | the controller works out bank, g, thrust | you |
| why | **this is the interface your policy uses** | to see what happens underneath |

`tab` hands the aircraft over live. Roll it inverted into a dive with the stick,
press `tab`, and watch guidance recover it — that is roughly the attitude an
untrained policy produces.

## Keys

Letters mirror the numpad 3×3, so it works one-handed and works on a laptop.

```
   Q W E          7 8 9          rudder / pitch / rudder
   A S D    ==    4 5 6          left / centre / right
   Z X C          1 2 3          throttle+ / pitch / throttle-
```

| axis | keys | **GUIDANCE** | **STICK** |
|---|---|---|---|
| left/right | `A` `D` | target heading ∓ | roll left / right |
| up | `W` | **climb** | **nose down** (stick forward) |
| down | `X` | descend | nose up |
| fore/aft | `Z` `C` | target speed ± | throttle up / down |
| rudder | `Q` `E` | — | rudder left / right |
| centre | `S` / space | freeze all three targets | centre the stick |
| `shift` | | bigger step | full deflection |

Arrow keys, `Home`/`PgUp`/`End`/`PgDn`/`Ins`/`Del` and the numpad all work too,
NumLock on or off.

Also: `tab` layer · `m` TARGET↔ACTION · `p` pause · `r` reset · `esc` quit ·
`+` `−` zoom · `g` trail length

> **⚠ The vertical axis means opposite things in the two layers.** `W` climbs in
> GUIDANCE and pushes the nose *down* in STICK, because autopilot convention
> ("up = up") and stick convention ("push = down") genuinely disagree. The first
> help line on screen always tells you which one is live: green `W/8 CLIMB` or
> orange `W/8 NOSE DOWN`.

### `m` — continuous vs discrete

Two grains of the same command, so you can feel what each kind of action space
is like. Fly a few minutes in both.

- **CONTINUOUS** (default) — keys nudge the setpoint finely; it stays where you
  leave it, like a cruise-control dial
- **DISCRETE** — the grid a policy gets: heading ±30°, speed ±20 kt, altitude
  ±1000 ft. Tap `D` → turns 30° and stops. Hold `D` → never stops turning

The gym exposes the discrete one, `Discrete(27)`.

## What the panel is telling you

| | |
|---|---|
| `turn cap` | best turn rate at **this speed and this altitude**. The purple tick is what you would get at 5,000 ft — that gap is what your altitude costs you |
| `ENERGY` | `Eh = h + V²/2g`. Blue is height, orange is speed. Blue shrinking while orange grows is a trade, not a loss; the bar getting shorter than the white 5-second mark is a loss |
| `Ps` | rate of change of `Eh`. **This is the bill for a hard turn** |
| `Nz` | current g, against what this speed and altitude can actually make |
| `alpha` / `authority` | STICK only. The FLCS cuts your pitch command as AoA rises — `authority 51 %` means half your stick is being thrown away, on purpose, to keep you out of a stall |

## Flying against the target

```bash
python -m aircombat_gym.tools.manual_operation --tacview --enemy circler
```

It orbits at 20,000 ft and 325 kt and does not shoot back.

You hit when you are **within 15° of the aim point, at 150–1,500 m, for 0.6 s**.
The aim point sits ahead of the target, not on it — **the pipper shows you where.**
Closer and squarer to his tail does more damage; a good shot kills in about 2 s.
`IN ZONE` means inside the envelope, `HITTING` means rounds are landing.
