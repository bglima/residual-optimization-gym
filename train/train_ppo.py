import gym

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import residual_optimization
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import torch as th

# Parallel environments
env = make_vec_env("SineCollisionEnv-v0", n_envs=4)
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

total_timesteps = 10_000
print(f'[INFO] Learning for {total_timesteps} timesteps.')

model = PPO("MlpPolicy", env, verbose=1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=1,
                             deterministic=True, render=False)

model.learn(total_timesteps=total_timesteps, callback=eval_callback, eval_freq=1_000)

filepath = os.path.abspath(os.path.dirname(__file__))
filename = datetime.datetime.now().strftime("ppo_model_%Y_%m_%d-%I:%M:%S_%p")
finalpath = os.path.join(filepath, '../', 'models', filename)
print('Saving model to ', finalpath)
model.save(finalpath)