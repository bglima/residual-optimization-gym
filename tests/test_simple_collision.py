from residual_optimization.envs.simple_collision_env import SimpleCollisionEnv
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
x_e = 0.1
K_e = 1000.0

# Trajectory definition
time = np.arange(start=time_start, stop=time_stop, step=dt)
num_samples = len(time)
# x_d = np.linspace(x_start, x_stop, num=num_samples, dtype=np.float64)
x_d = np.ones_like(time, dtype=np.float64) * 0.2
f_d = np.ones_like(time, dtype=np.float64) * 5.0

x_c = np.zeros_like(x_d, dtype=np.float64)
f_e = np.zeros_like(f_d, dtype=np.float64)

# Environment
# x_e = 0.05
# K_e = 1
env = SimpleCollisionEnv(
    base_controller=naive_controller,
    dt=dt,
    x_e=np.array([x_e], dtype=np.float64),
    K_e=np.array([K_e], dtype=np.float64)
)
observation = env.reset()

# Main loop
x_o = 0
for t in range(len(time)):
    # Update controllers    
    naive_controller.set_reference(np.array([x_d[t], f_d[t]], dtype=np.float64))   # Setpoint for u_h in form [x_d, f_d]
    action = env.action_space.sample()          # Sample random u_r
    obs, reward, done, info = env.step(action)  # Calculate reward

    # Plot
    x_o = info['x_o']
    # print(f'Step {t}, x_o: {x_o}, x_d_diff: {x_d_diff}, f_d: {obs[0]}, f_e: {obs[1]}, reward: {reward}, done: {done}')

    # Add to plot
    x_c[t] = x_o
    f_e[t] = obs[1]

# Plot variables
plt.figure()
ax1 = plt.subplot(211)
ax1.axhline(y=0.1, color='r', linestyle='-')
ax1.plot(time, x_d, color='blue', linestyle='--')
ax1.plot(time, x_c, color='black')
ax1.set_ylabel('Position (m)')
ax1.set_title('Time (sec)')
ax1.set_ylim([0, 0.3])

ax2 = plt.subplot(212)
ax2.plot(time, f_d, color='blue', linestyle='--')
ax2.plot(time, f_e, color='black')
ax2.set_ylabel('Force (N)')
plt.xlabel("Time (sec)")
plt.show()