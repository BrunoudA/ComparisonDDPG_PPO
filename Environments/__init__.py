from gym.envs.registration import register


register(
    id="Crosswalk_comparison-v0",
    entry_point="Environments.Env_comparison:Crosswalk_comparison",
    max_episode_steps=90,
    reward_threshold=20.0,
)

register(
    id="Crosswalk_test-v0",
    entry_point="Environments.Env_test:Crosswalk_test",
    max_episode_steps=90,
    reward_threshold=20.0,
)