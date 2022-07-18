import numpy as np
import gym

from residual_optimization.controllers.base_controller import BaseController

class SimpleCollisionEnv(gym.Env):
    """
    This follows the gym.Env API. For a detailed guide, check:
    https://www.gymlibrary.ml/content/api/

    This environment simulates an end-effector in collision with an environment
    (i.e. a wall) to be used along with an interaction controller. Assume that
    gravity may be discarded (i.e. low-level compensated), and that the robot
    performs a straight line towards the environment along a single axis Z.

    Our model of the control variable "u" is defined as follows:
        u = u_h + u_r
    where
        "u_h" : the desired control output given by a simplified model "base_controller"
        "u_r" : the residual unknown force that is given back by the environment

    The simplified model "base_controller" is passed as an argument of the constructor

    Our policy needs to search for "u_r", comprising our action space.
    Our action "u_r" is function of two observation variables, which are:
        "f_d" : the desired force to be applied by the end-effector
        "f_e" : the force which is applied back by the environment

    Our reward is (TODO).
    """
    def __init__(self, base_controller: BaseController):
        super(SimpleCollisionEnv, self).__init__()
        self.base_controller = base_controller

        # Define our action space for "u_r"
        self.action_shape = (1,)
        self.action_min = -1
        self.action_max = 1

        # We use Box since our neural network residual action is continuous
        self.action_space = gym.spaces.Box(
            low=self.action_min,
            high=self.action_max,
            shape=self.action_shape
        )

        # Our observation space are two forces: "f_d" and "f_e"
        self.observation_shape = (2, 3)
        self.observation_min = -50
        self.observation_max = 50
        self.observation_space = gym.spaces.Box(
            low=self.observation_min,
            high=self.observation_max,
            shape=self.observation_shape
        )

        # Initial state for our observation variable
        self.state = self.observation_space.sample()

        # Maximum difference between "f_d" and "f_e" so that our episode is considered a success
        self.tolerance = 1e-3

        # Total number of actions before terminal state
        self.steps = 0
        self.max_steps = int(1e3)

    def step(self, action):
        """
        Receives the neural network action, sums up with the base_controller action
        and apply the sum to the environment.
        """
        # Check if the neural network action is within its bounds
        assert self.action_space.contains(action), f"{action} is not a valid action."

        self.steps += 1

        # Gets the simple controller action
        # Currently, the base_controller state is the same as the neural network one
        base_action = self.base_controller.update(self.state, 0.01)

        # Sums the action from the neural network policy with the simple controller action
        u = base_action + action

        # TODO: Model the environment response given the summed action; For now,
        # return a random observation.
        observation = self.observation_space.sample()

        # Calculate the error given by "f_d" - "f_e"
        error = observation[0] - observation[1]

        # TODO: Model the reward based on the observations
        reward = 1 if np.max(error) < self.tolerance else -1

        # Check if within limits
        done = False
        if (self.steps > self.max_steps):
            done = True

        info = {
            'state': np.array(self.state),
            'action': action,
            'done': done,
            'reward': reward
        }

        return observation, reward, done, info

    def reset(self):
        self.steps = 0
        self.state = self.observation_space.sample()
        return self.state

