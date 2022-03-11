from gym.envs.registration import register


register(
    id="Crosswalk_comparison-v0",
    entry_point="Environments.Env_comparison:Crosswalk_comparison",
    max_episode_steps=90,
    reward_threshold=20.0,
)

register(
    id="Crosswalk_comparison2-v0",
    entry_point="Environments.Env_comparison2:Crosswalk_comparison2",
    max_episode_steps=90,
    reward_threshold=20.0,
)

register(
    id="Crosswalk_comparisonQP-v0",
    entry_point="Environments.Env_comparison_QP:Crosswalk_comparison_QP",
    max_episode_steps=90,
    reward_threshold=20.0,
)
register(
    id="Crosswalk_comparisonDDPG-v0",
    entry_point="Environments.Env_comparison_DDPG:Crosswalk_comparison_DDPG",
    max_episode_steps=90,
    reward_threshold=20.0,
)