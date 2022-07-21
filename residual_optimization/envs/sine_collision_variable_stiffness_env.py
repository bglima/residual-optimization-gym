import numpy as np
import gym

from residual_optimization.controllers.base_controller import BaseController

class SineCollisionVariableStiffnessEnv(gym.Env):
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

    The human-provided model "base_controller" is passed as an argument to the constructor.

    Our policy needs to search for "u_r", comprising our action space.

    Our action "u_r" is function of two observation variables, which are:
        "f_d" : the desired force to be applied by the agente end-effector
        "f_e" : the force which is applied back by the environment

    Internal parameters
    -------------------
    x_e : float
        The position in meters of the wall with respect to the origin.
    K_e : float
        The stiffiness of the environment.

    Reward
    ------
    Initially it is defined as a shaped reward which is the negative squared
    error
        r = -(f_d - f_e) ** 2
    """
    def __init__(self,
        base_controller : BaseController,
        dt : np.float64,
        K_e_1 : np.float64,
        K_e_2 : np.float64,
        K_e_toggle_time : np.float64,
        x_e_amplitude : np.float64,
        x_e_offset : np.float64,
        x_e_frequency : np.float64
    ):
        """
        This environment has the wall position varying with a fixe offset and amplitude.

        Parameters
        ----------
        base_controller : controllers.BaseController
            It is the human-defined controller given as the base controller.
        dt : float
            It is the duratin of the control cycle for both the human-defined and 
            the learned policy controller.
        K_e_1 : float
            The first value of stiffness for the environment
        K_e_2 : float
            The second value of stiffness for the environment
        K_e_toggle : float
            The time in seconds in which the stiffness will 
        x_e_amplitude : float
            The amplitude of the environment position sinusoid
        x_e_offset : float
            The offset of the sinusoid from the beggining
        x_e_frequency : float
            The frequency is hz of the sinusoid
        """
        super(SineCollisionVariableStiffnessEnv, self).__init__()
        self.base_controller = base_controller
        self.dt = dt
        self.time = 0

        # Define the starting position for the robot
        self.x_o_shape = (1,)
        self.x_o_min = 0
        self.x_o_max = 0
        self.x_o_box = gym.spaces.Box(
            low=self.x_o_min,
            high=self.x_o_max,
            shape=self.x_o_shape,
            dtype=np.float64
        )
        self.x_o = self.x_o_box.sample()

        # Define the environment contact position x_e
        self.x_e = 0
        self.x_e_amplitude = x_e_amplitude
        self.x_e_frequency = x_e_frequency
        self.x_e_offset = x_e_offset
        self.K_e = 0
        self.K_e_1 = K_e_1
        self.K_e_2 = K_e_2
        self.K_e_toggle_time = K_e_toggle_time

        # Define our action space for "u_r"
        self.action_shape = (1,)
        self.action_min = -0.01
        self.action_max = 0.01

        # We use Box since our neural network residual action is continuous
        self.action_space = gym.spaces.Box(
            low=self.action_min,
            high=self.action_max,
            shape=self.action_shape,
            dtype=np.float64
        )

        # Our observation space are two uni-dimensional forces: "f_d" and "f_e"
        self.observation_shape = (2, 1)
        self.observation_min = -50
        self.observation_max = 50
        self.observation_space = gym.spaces.Box(
            low=self.observation_min,
            high=self.observation_max,
            shape=self.observation_shape,
            dtype=np.float64
        )
        self.observation = np.zeros(shape=(2,1))

        # Maximum difference between "f_d" and "f_e" so that our episode is considered a success
        self.tolerance = 1e-3

        # Total number of actions before terminal state
        self.steps = 0
        self.max_steps = int(1e3)

    def step(self, action):
        """
        Receives the neural network action, sums up with the base_controller action
        and apply the sum to the environment.

        Parameters
        ----------
        action : np.ndarray
            The control variable "u_r" output by the policy. It is defined as the residual
            of the next position in end-effector frame.
        """
        assert self.action_space.contains(action), f"{action} is not a valid policy action."
        self.steps += 1
        self.time += self.dt

        # Calculate human-designed control u_h
        # and policy control u_r
        _, f_e = self.observation
        u_h = self.base_controller.update(f_e, self.dt)
        u_r = action
        
        # Uncomment next line to allow the policy addition
        # u = u_h + u_r
        u = u_h

        # Update absolute pose
        self.x_o = u

        # Define K_e based on time
        if self.time > self.K_e_toggle_time:
            self.K_e = self.K_e_2
        else:
            self.K_e = self.K_e_1
        
        # Get environment response
        self.x_e = self.x_e_offset + self.x_e_amplitude * np.sin(2 * np.pi * self.x_e_frequency * self.time)
        if self.x_o <= self.x_e:
            f_e = np.array([0.0], dtype=np.float64)
        else:
            f_e = np.multiply(self.K_e, self.x_o - self.x_e) + np.random.normal(0, 0.1)
            
        # Calculate the error given by "f_d" - "f_e"
        _, f_d = self.base_controller.get_reference()
        d_f = f_d - f_e
        self.observation = np.hstack((f_d, f_e)).reshape(2,1)

        # Calculate the reward
        reward = -d_f ** 2

        # Check if within limits
        done = False
        if (self.steps > self.max_steps):
            done = True

        info = {
            'x_o' : self.x_o,
            'x_e' : self.x_e
        }

        return self.observation, reward, done, info

    def reset(self):
        self.x_o = self.x_o_box.sample()
        self.x_e = 0
        self.steps = 0
        self.time = 0
        self.K_e = self.K_e_1
        return np.array([0.0, 0.0])

