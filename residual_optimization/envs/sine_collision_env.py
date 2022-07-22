import numpy as np
import gym
import matplotlib.pyplot as plt

from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D

class SineCollisionEnv(gym.Env):
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
    base_controller : controllers.BaseController
        It is the human-defined controller given as the base controller.

    Reward
    ------
    Initially it is defined as a shaped reward which is the negative squared
    error
        r = -(f_d - f_e) ** 2
    """
    def __init__(self,
        dt : np.float64 = 0.01,
        K_e : np.float64 = 1000,
        x_e_amplitude : np.float64 = 0.015,
        x_e_offset : np.float64 = 0.1,
        x_e_frequency : np.float64 = 0.2,
        max_u_r : np.float64 = 0.01,
        # Base controller variables
        x_d_start : np.float64 = 0.,
        x_d_stop : np.float64 = 0.,
        f_d_start : np.float64 = 5.0,
        f_d_stop :  np.float64 = 5.0,
        time_start : np.float64 = 0,
        time_stop : np.float64 = 10,
        M_d_inv : np.ndarray = np.array([1], dtype=np.float64),
        K_P : np.ndarray = np.array([0.0], dtype=np.float64),
        K_D : np.ndarray = np.array([42.0], dtype=np.float64),
    ):
        """
        This environment has the wall position varying with a fixe offset and amplitude.

        Parameters
        ----------
        dt : float
            It is the duratin of the control cycle for both the human-defined and 
            the learned policy controller.
        K_e : float
            The stiffness of the environment
        x_e_amplitude : float
            The amplitude of the environment position sinusoid
        x_e_offset : float
            The offset of the sinusoid from the beggining
        x_e_frequency : float
            The frequency is hz of the sinusoid
        x_d : np.ndarray
            The desided positions
        f_d : np.ndarray
            The desired forces
        
        """
        print("Environment is being init.")
        super(SineCollisionEnv, self).__init__()

        # General purpose variables
        self.time = np.arange(start=time_start, stop=time_stop, step=dt)
        self.dt = dt
        self.num_samples = len(self.time)
        self.x_d = np.linspace(start=x_d_start, stop=x_d_stop, num=self.num_samples)
        self.f_d = np.linspace(start=f_d_start, stop=f_d_stop, num=self.num_samples)

        # Define the environment contact position x_e
        self.x_e_amplitude = x_e_amplitude
        self.x_e_frequency = x_e_frequency
        self.x_e_offset = x_e_offset
        self.K_e = K_e

        # Instantiate the base controller
        self.base_controller = AdmittanceController1D(
            M_d_inv = M_d_inv,
            K_P = K_P,
            K_D = K_D
        )

        # Define our action space for "u_r"
        self.max_u_r = max_u_r
        self.action_shape = (1,)
        self.action_min = -1.0
        self.action_max = 1.0
        self.action_space = gym.spaces.Box(
            low=self.action_min,
            high=self.action_max,
            shape=self.action_shape,
            dtype=np.float64
        )

        # Our observation space are "f_d", "f_e", and "u_h"
        self.observation_shape = (3, 1)
        self.observation_min = 0
        self.observation_max = 10
        self.observation_space = gym.spaces.Box(
            low=self.observation_min,
            high=self.observation_max,
            shape=self.observation_shape,
            dtype=np.float64
        )
        self.observation = np.zeros(shape=self.observation_shape)

        # Total number of actions before terminal state
        self.max_steps = len(self.x_d) - 1

    def reset(self):
        print("Environment is being reset.")
        self.base_controller.reset()
        self.x_o = 0.0
        self.x_e = 0.0
        self.steps = 0
        self.current_time = 0.0
        self.x_c = np.zeros_like(self.x_d, dtype=np.float64)
        self.x_e = np.zeros_like(self.x_d, dtype=np.float64)
        self.f_e = np.zeros_like(self.f_d, dtype=np.float64)
        self.u_h = np.zeros_like(self.f_d, dtype=np.float64)
        self.u_r = np.zeros_like(self.f_d, dtype=np.float64)
        self.last_ur = np.zeros(self.action_shape)
        self.u = np.zeros_like(self.f_d, dtype=np.float64)
        self.observation = np.zeros(self.observation_shape, dtype=np.float64)
        self.last_ur = np.zeros(self.action_shape, dtype=np.float64)
        return self.observation

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
        self.current_time += self.dt

        # Calculate human-designed control u_h
        # and policy control u_r
        _, f_e, u_h = self.observation
        u_h_reference = np.array([self.x_d[self.steps-1], self.f_d[self.steps-1]], dtype=np.float64)
        self.base_controller.set_reference(u_h_reference)   # Setpoint for u_h in form [x_d, f_d]
        
        u_h = self.base_controller.update(f_e, self.dt)
        
        # As action is in the range
        u_r = action * self.max_u_r
        
        # Uncomment next line to allow the policy addition
        u = u_h + u_r

        # Update absolute pose
        self.x_o = u

        # Get environment response
        x_e = self.x_e_offset + self.x_e_amplitude * np.sin(2 * np.pi * self.x_e_frequency * self.current_time)
        if self.x_o <= x_e:
            f_e = np.array([0.0], dtype=np.float64)
        else:
            f_e = np.multiply(self.K_e, self.x_o - x_e) + np.random.normal(0, 0.1)
            
        # Calculate the error given by "f_d" - "f_e"
        _, f_d = self.base_controller.get_reference()
        d_f = f_d - f_e
        self.observation = np.hstack((f_d, f_e, u_h)).reshape( self.observation_shape )

        # Calculate the reward
        reward = -0.01 * np.linalg.norm(d_f) - \
                    10 * np.linalg.norm(u_r) - \
                     1 * np.linalg.norm( self.last_ur - u_r )

        # Check if within limits
        done = False
        if (self.steps >= self.max_steps):
            done = True

        info = {
            'x_o' : self.x_o,
            'x_e' : x_e,
            'u_h' : u_h,
            'u_r' : u_r,
            'u' : u
        }

        return self.observation, reward, done, info