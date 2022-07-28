from residual_optimization.envs.sine_collision_stiffness_estimator_env import SineCollisionStiffnessEstimator
from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D
import numpy as np

import matplotlib.pyplot as plt
import cvxpy as cp

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_ylabel('Abs Error (N)')
ax1.set_xlabel('Time (s)')

ax1.title.set_text('Absolute Force-Tracking Errors per Approach')

linewidth = 0.7

configs = [
    {
        'title': 'admittance_only',
        'alpha': 0,
        'beta': 0,
        'testing': True,
        'color': 'blue',
    },
    {
        'title': 'admittance_plus_residual',
        'alpha': 0,
        'beta': 0,
        'testing': False,
        'color': 'green',
    },
    {
        'title': 'admittance_plus_regularized_residual',
        'alpha': 0.1,
        'beta': 10,
        'testing': False,
        'color': 'red',
    },
]

for arg in configs:
    np.random.seed(42)

    # Gym environment
    env = SineCollisionStiffnessEstimator(
        testing=arg['testing'],
        alpha=arg['alpha'],
        beta=arg['beta'],
        time_start=1,
        time_stop=5,
        K_e_tilde_std=500,
        dt=0.01,
        solver='OSQP'
    )

    # Trajectory definitions
    time = env.time
    num_samples = len(time)
    x_d = env.x_d
    f_d = env.f_d
    x_c = np.zeros_like(x_d, dtype=np.float64)
    x_e = np.zeros_like(x_d, dtype=np.float64)
    f_e = np.zeros_like(f_d, dtype=np.float64)
    d_f = np.zeros_like(f_d, dtype=np.float64)
    abs_d_f = np.zeros_like(f_d, dtype=np.float64)
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
        d_f[t] = f_d[t] - f_e[t]
        abs_d_f[t] = abs( d_f[t] )
    
    print('- ', arg['title'])
    # Excluding moments without environment sensing
    nonzero_indexes = np.nonzero(f_e)
    non_zero_abs_d_f = abs_d_f[nonzero_indexes]

    # Printing values
    print('Max abs_d_f: ', non_zero_abs_d_f.max())
    print('Min abs_d_f: ', non_zero_abs_d_f.min())
    print('Mean abs_d_f: ', non_zero_abs_d_f.mean())
    print('Std abs_d_f: ', non_zero_abs_d_f.std())
    print('Integral of abs_d_f: ', non_zero_abs_d_f.sum())
    print('\n')

    ax1.plot(time, abs_d_f, color=arg['color'], label='{}'.format(arg['title']), linewidth=linewidth)

ax1.legend()
plt.show()

