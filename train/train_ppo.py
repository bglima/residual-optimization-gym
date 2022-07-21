import gym

from stable_baselines3 import PPO
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
total_episodes = 10

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

model.save("training_sine_constant_stiffness_1000ep")

# Visualization
obs = env.reset()
for t in range(len(time)):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    
    # Add to plot
    x_c[t] = info['x_o']
    x_e[t] = info['x_e']
    u_h[t] = info['u_h']
    u_r[t] = info['u_r']
    u[t] = info['u']
    f_e[t] = obs[1]

# Plot variables
fig = plt.figure()
ax1 = plt.subplot(311)
linewidth = 0.7
ax1.plot(time, x_d, color='blue', linestyle='--', label='x_d', linewidth=linewidth)
ax1.plot(time, x_c, color='black', label='x_c', linewidth=linewidth)
ax1.plot(time, x_e, color='red', linestyle='--', label='x_e', linewidth=linewidth)
ax1.set_ylabel('Position (m)')
ax1.title.set_text('Position tracking')
ax1.set_ylim([0, 0.22])
ax1.legend()

ax2 = plt.subplot(312)
ax2.plot(time, f_d, color='blue', linestyle='--', label='f_d', linewidth=linewidth)
ax2.plot(time, f_e, color='black', label='f_e', linewidth=linewidth)
ax2.set_ylabel('Force (N)')
ax2.legend()
ax2.title.set_text('Force tracking')

ax3 = plt.subplot(313)
ax3.plot(time, u_h, color='red', linestyle='-', label='u_h', linewidth=linewidth)
ax3.plot(time, u_r, color='blue', linestyle='-', label='u_r', linewidth=linewidth)
ax3.plot(time, u, color='black', linestyle='-', label='u', linewidth=linewidth)
ax3.set_ylabel('Policy Action (m)')
ax3.legend()
ax3.title.set_text('Force tracking')

plt.xlabel("Time (sec)")

fig.subplots_adjust(hspace=0.5)
plt.show()