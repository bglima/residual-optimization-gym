from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# You need to import the package so that you can use the env id
import residual_optimization

# Parallel environments
env = make_vec_env("SineCollisionEnv-v0", n_envs=4)