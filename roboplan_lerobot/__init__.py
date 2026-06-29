from gymnasium.envs.registration import register

register(
    id="roboplan_lerobot/KinovaReach-v0",
    entry_point="roboplan_lerobot.envs:KinovaEnv",
    max_episode_steps=500,
)
