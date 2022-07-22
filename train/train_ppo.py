import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import residual_optimization
from residual_optimization.envs.sine_collision_env import SineCollisionEnv
from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import torch as th

# Parallel environments
env = make_vec_env("SineCollisionEnv-v0", n_envs=4)
model = PPO("MlpPolicy", env, verbose=1)

total_timesteps = 10_000
print(f'[INFO] Learning for {total_timesteps} timesteps.')

model.learn(total_timesteps=total_timesteps)

filepath = os.path.abspath(os.path.dirname(__file__))
filename = datetime.datetime.now().strftime("ppo_model_%Y_%m_%d-%I:%M:%S_%p")
finalpath = os.path.join(filepath, '../', 'models', filename)
print('Saving model to ', finalpath)
model.save(finalpath)