import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
from residual_optimization.envs.sine_collision_env import SineCollisionEnv
from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D
import numpy as np
import matplotlib.pyplot as plt

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

# Gym environment
env = SineCollisionEnv(
    base_controller=naive_controller,
    dt=dt,
    x_e_offset=x_e_offset,
    x_e_amplitude=x_e_amplitude,
    x_e_frequency=x_e_frequency,
    K_e=np.array([K_e], dtype=np.float64)
)

model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=num_samples * total_episodes)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()



# Main loop
x_o = 0
for t in range(len(time)):
    # Update controllers    
    naive_controller.set_reference(np.array([x_d[t], f_d[t]], dtype=np.float64))   # Setpoint for u_h in form [x_d, f_d]
    action = env.action_space.sample()          # Sample random u_r
    obs, reward, done, info = env.step(action)  # Calculate reward

    # Add to plot
    x_c[t] = info['x_o']
    x_e[t] = info['x_e']
    f_e[t] = obs[1]