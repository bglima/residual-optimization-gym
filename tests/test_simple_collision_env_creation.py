from residual_optimization.envs.simple_collision_env import SimpleCollisionEnv
from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D
import numpy as np

naive_controller = AdmittanceController1D(
    M_d = np.array([1.0], dtype=np.float64),
    K_P = np.array([100.0], dtype=np.float64),
    K_D = np.array([0.0], dtype=np.float64)
)

env = SimpleCollisionEnv(
    base_controller=naive_controller,
    dt=0.01
)

observation = env.reset()
env.max_steps = 50

for step in range(env.max_steps + 1):
    naive_controller.set_reference(np.array([0.01, 0], dtype=np.float64))   # Setpoint for u_h in form [x_d, f_d]

    action = env.action_space.sample()          # Sample random u_r

    obs, reward, done, info = env.step(action)  # Calculate reward

    x_o = info['x_o']

    print(f'Step {step}, x_o: {x_o}, f_d: {obs[0]}, f_e: {obs[1]}, reward: {reward}, done: {done}')
