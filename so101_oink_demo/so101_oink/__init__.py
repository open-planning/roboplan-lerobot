"""SO-101 pick-cube simulation driven by RoboPlan's OInK solver."""

from gymnasium.envs.registration import register

from so101_oink.env import SO101PickCubeEnv  # noqa: F401

register(id="SO101PickCube-v0", entry_point="so101_oink.env:SO101PickCubeEnv", max_episode_steps=500)
