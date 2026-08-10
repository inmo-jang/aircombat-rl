"""Shared simulation: the aircraft and how it is flown.

Everything above this -- tasks, baselines, tools -- consumes it and none of it
knows which task is running.  A BVR scenario reuses all of it unchanged; only
the weapon model and the observation change.
"""
