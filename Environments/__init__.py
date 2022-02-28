from gym.envs.registration import register


register(
    id="Crossway_comparison-v0",
    entry_point="Environments.Env_comparison:Crosswalk_comparison",
    max_episode_steps=90,
    reward_threshold=20.0,
)