from gym.envs.registration import register


register(
    id="Crossway_comparison-v0",
    entry_point="Environments.Env_comparison:Crosswalk_comparison",
    max_episode_steps=90,
    reward_threshold=20.0,
)

register(
    id="Crossway_comparison2-v0",
    entry_point="Environments.Env_comparison2:Crosswalk_comparison2",
    max_episode_steps=90,
    reward_threshold=20.0,
)