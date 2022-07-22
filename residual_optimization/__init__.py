from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

# Hook to load plugins from entry points
_load_env_plugins()

register(
    id="SineCollisionEnv-v0",
    entry_point="residual_optimization.envs.sine_collision_env:SineCollisionEnv",
    max_episode_steps=1000
)