from residual_optimization.envs.sine_collision_stiffness_estimator_env import SineCollisionStiffnessEstimator
from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D
import numpy as np

import matplotlib.pyplot as plt
import cvxpy as cp

solvers = ['OSQP', 'ECOS', 'SCS']
colors = ['blue', 'orange', 'green']
solver_f_e = []
solver_u_r = []

# Plot variables
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_ylabel('Force (N)')
ax1.legend()
ax1.title.set_text('Force tracking')

ax2.set_ylabel('Residual Control Action (m)')
ax2.legend()
ax2.title.set_text('Optimization Variable')

linewidth = 0.7

for i in range(len(solvers)):
    print(f'Solver {i}: {solvers[i]}')
    # If we comment the seed, solutions should be different
    np.random.seed(42)

    # Gym environment
    env = SineCollisionStiffnessEstimator(
        testing=False,
        alpha=0.1,
        beta=10,
        time_stop=5,
        K_e_tilde_std=500,
        dt=0.005,
        solver=solvers[i]
    )

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

    nonzero_indexes = np.nonzero(env.residual_controller_durations)
    nonzero_residual_controller_durations = env.residual_controller_durations[nonzero_indexes:-1]
    
    print('[INFO] Avg duration for residual controller: {:.4f}'.format(nonzero_residual_controller_durations.mean()))
    print('[INFO] Std duration for residual controller: {:.4f}'.format(nonzero_residual_controller_durations.std()))
    print('[INFO] Max duration for residual controller: {:.4f}'.format(nonzero_residual_controller_durations.max()))
    print('[INFO] Min duration for residual controller: {:.4f}'.format(nonzero_residual_controller_durations.min()))
    
    ax1.plot(time, f_e, color=colors[i], label=f'f_e_{solvers[i]}', linewidth=linewidth)
    ax2.plot(time, u_r, color=colors[i], linestyle='-', label=f'u_r_{solvers[i]}', linewidth=linewidth)


plt.ylabel('sec / iteration')
plt.xlabel('Time (sec)')
plt.legend()
plt.xlabel(f"Benchmark over {num_samples} iterations")   
plt.show()
