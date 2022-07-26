from residual_optimization.envs.sine_collision_stiffness_estimator_env import SineCollisionStiffnessEstimator
import numpy as np
import matplotlib.pyplot as plt

# Gym environment
env = SineCollisionStiffnessEstimator(testing=False, alpha=0.1, beta=10, time_stop=5, K_e_tilde_std=500)

# Trajectory definitions
time = env.time
num_samples = len(time)
x_d = env.x_d
f_d = env.f_d
x_c = np.zeros_like(x_d, dtype=np.float64)
x_e = np.zeros_like(x_d, dtype=np.float64)
f_e = np.zeros_like(f_d, dtype=np.float64)
u_h = np.zeros_like(f_d, dtype=np.float64)
u_r = np.zeros_like(f_d, dtype=np.float64)
u = np.zeros_like(f_d, dtype=np.float64)

# Visualization
obs = env.reset()
for t in range(len(time)):
    obs, info = env.step()
    
    # Add to plot
    x_c[t] = info['x_o']
    x_e[t] = info['x_e']
    u_h[t] = info['u_h']
    u_r[t] = info['u_r']
    u[t] = info['u']
    f_e[t] = info['f_e']

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
# ax3.plot(time, u_h, color='red', linestyle='-', label='u_h', linewidth=linewidth)
ax3.plot(time, u_r, color='blue', linestyle='-', label='u_r', linewidth=linewidth)
# ax3.plot(time, u, color='black', linestyle='-', label='u', linewidth=linewidth)
ax3.set_ylabel('Residual Control Action (m)')
ax3.legend()
ax3.title.set_text('Optimization Variable')

plt.xlabel("Time (sec)")

fig.subplots_adjust(hspace=0.5)
plt.show()
