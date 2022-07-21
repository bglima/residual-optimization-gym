import gym

from stable_baselines3 import PPO
from residual_optimization.envs.sine_collision_env import SineCollisionEnv
from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

naive_controller = AdmittanceController1D(
    M_d_inv = np.array([1], dtype=np.float64),
    K_P = np.array([0.0], dtype=np.float64),
    K_D = np.array([42.0], dtype=np.float64)
)

# Trajectory variables
dt = 0.01
time_start = 0
time_stop = 10
x_start = 0
x_stop = 0.2
x_e_offset = 0.1
x_e_amplitude = 0.015
x_e_frequency = 0.2
K_e = 1000.0
total_episodes = 100

# Trajectory definitions
time = np.arange(start=time_start, stop=time_stop, step=dt)
num_samples = len(time)
x_d = np.ones_like(time, dtype=np.float64) * 0.2
f_d = np.ones_like(time, dtype=np.float64) * 5.0
x_c = np.zeros_like(x_d, dtype=np.float64)
x_e = np.zeros_like(x_d, dtype=np.float64)
f_e = np.zeros_like(f_d, dtype=np.float64)
u_h = np.zeros_like(f_d, dtype=np.float64)
u_r = np.zeros_like(f_d, dtype=np.float64)
u = np.zeros_like(f_d, dtype=np.float64)

# Gym environment
env = SineCollisionEnv(
    base_controller=naive_controller,
    dt=dt,
    x_e_offset=x_e_offset,
    x_e_amplitude=x_e_amplitude,
    x_e_frequency=x_e_frequency,
    K_e=np.array([K_e], dtype=np.float64),
    x_d=x_d,
    f_d=f_d
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=num_samples * total_episodes)

filepath = os.path.abspath(os.path.dirname(__file__))
filename = datetime.datetime.now().strftime("ppo_model_%Y_%m_%d-%I:%M:%S_%p")
finalpath = os.path.join(filepath, '../', 'models', filename)
print('Saving model to ', finalpath)
model.save(finalpath)