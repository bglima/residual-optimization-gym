from residual_optimization.envs.simple_collision_env import SimpleCollisionEnv
from residual_optimization.controllers.simple_controller import SimpleController

base_controller = SimpleController(
    observation_shape=(2,3),    # x_d and f_d
    observation_min=-1,
    observation_max=-1,
    action_shape=(1,),          # control variable u_h
    action_min=-1,
    action_max=1
)

env = SimpleCollisionEnv(
    base_controller=base_controller
)

observation = env.reset()

for step in range(env.max_steps + 1):
    print("Step {}".format(step))

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    print(info)