"""Air combat environments on JSBSim's F-16.

    import gymnasium as gym
    import aircombat_gym.wvr.envs        # registers the ids

    env = gym.make("AirCombat/Circular-v0")

Nothing here imports a learning library, and nothing here computes a reward --
the environment hands out the material and the student writes the function.
"""
