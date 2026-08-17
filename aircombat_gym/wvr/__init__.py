"""Within visual range: the gun.

`engagement` is the weapon.  `obs` and `actions` are the two halves of the
interface a policy binds to -- what it sees and what it may ask for -- and
`baselines` holds the scripted opponents written against them.  `envs` assembles
those into the Gym ids a task registers.

All of it is specific to *this* scenario: 1v1, guns.  A `bvr` package sits
beside this one when missiles arrive, with its own weapon, its own observation
and its own bots.  The aircraft and the autopilot underneath do not change.
"""
