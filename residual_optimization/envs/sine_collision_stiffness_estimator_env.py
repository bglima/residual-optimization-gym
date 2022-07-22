import numpy as np
import gym
import matplotlib.pyplot as plt
import cvxpy as cp

from residual_optimization.controllers.admittance_controller_1d import AdmittanceController1D

class SineCollisionStiffnessEstimator(gym.Env):
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
        K_e_tilde_mean : np.float64 = 0,
        K_e_tilde_std : np.float64 = 100,
        x_e_amplitude : np.float64 = 0.015,
        x_e_offset : np.float64 = 0.1,
        x_e_frequency : np.float64 = 0.2,
        max_u_r : np.float64 = 0.01,
        # Base controller variables
        x_d_start : np.float64 = 0.2,
        x_d_stop : np.float64 = 0.2,
        f_d_start : np.float64 = 5.0,
        f_d_stop :  np.float64 = 5.0,
        time_start : np.float64 = 0,
        time_stop : np.float64 = 10,
        M_d_inv : np.ndarray = np.array([1], dtype=np.float64),
        K_P : np.ndarray = np.array([0.0], dtype=np.float64),
        K_D : np.ndarray = np.array([42.0], dtype=np.float64),
        testing : bool = False,
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
        super(SineCollisionStiffnessEstimator, self).__init__()

        # General purpose variables
        self.testing = testing
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
        self.K_e_tilde_mean = K_e_tilde_mean
        self.K_e_tilde_std = K_e_tilde_std
        self.f_e = 0
        # Instantiate the base controller
        self.base_controller = AdmittanceController1D(
            M_d_inv = M_d_inv,
            K_P = K_P,
            K_D = K_D
        )

        # Define our action space for "u_r"
        self.u_r = cp.Variable()
        self.max_u_r = max_u_r
        self.action_shape = (1,)
        self.observation_shape = (1,)
        self.x_o = 0.0
        self.x_o_prev = 0.0
        self.x_o_dot = 0.0

        # Total number of actions before terminal state
        self.max_steps = len(self.x_d) - 1

    def get_optimal_action(self, obs : np.ndarray):
        """
        Perform a optimization over d_f in order to find the best
        residual action u_r_star. Receives as observation the desired
        and the ference force [f_e_hat, f_d]

        Parameters
        ----------
            obs : [f_e_hat, f_d]
                Where f_e_hat is the 1d-array containing the estimated 
                force from the environment, and f_d is the 1d-array 
                with the force reference for the admittance controller
        """
        # TODO: Perform the u_r_start calculation
        u_r_star = np.zeros(shape=self.action_shape)
        return u_r_star

    def reset(self):
        self.base_controller.reset()
        self.x_o = 0.0
        self.x_e = 0.0
        self.steps = 0
        self.current_time = 0.0
        self.observation = np.zeros(self.observation_shape, dtype=np.float64)
        self.last_ur = np.zeros(self.action_shape, dtype=np.float64)
        return self.observation

    def get_K_e_hat(self):
        K_e_tilde = np.random.normal(self.K_e_tilde_mean, self.K_e_tilde_std)
        K_e_hat = self.K_e + K_e_tilde
        return K_e_hat

    def step(self):
        """
        Receives the neural network action, sums up with the base_controller action
        and apply the sum to the environment.

        Parameters
        ----------
        action : np.ndarray
            The control variable "u_r" output by the policy. It is defined as the residual
            of the next position in end-effector frame.
        """
        self.steps += 1
        self.current_time += self.dt

        # Calculate human-designed control u_h
        # and policy control u_r
        f_d = self.f_d[self.steps-1]
        x_d = self.x_d[self.steps-1]
        u_h_reference = np.array([x_d, f_d], dtype=np.float64)
        self.base_controller.set_reference(u_h_reference)   # Setpoint for u_h in form [x_d, f_d]
        
        u_h = self.base_controller.update((self.f_e, self.x_o, self.x_o_dot), self.dt)
                  
        # Get environment response
        self.x_e = self.x_e_offset + self.x_e_amplitude * np.sin(2 * np.pi * self.x_e_frequency * self.current_time)
        if self.f_e == 0:
            u_r_star = self.max_u_r
        else:
            K_e_hat = self.get_K_e_hat()
            x_e_hat = self.x_o - (K_e_hat ** -1) * self.f_e

            constraints = [
                self.u_r >= -self.max_u_r,
                self.u_r <= self.max_u_r
            ]

            obj = cp.Minimize( (K_e_hat * (u_h + self.u_r - x_e_hat) - f_d ) ** 2)
            
            prob = cp.Problem(obj, constraints)
            prob.solve()  # Returns the optimal value.
            u_r_star = self.u_r.value

        # Uncomment next line to allow the policy addition
        if (self.testing):
            u = u_h
        else:
            u = u_h + u_r_star

        # Update absolute pose
        self.x_o = u
        if self.x_o <= self.x_e:
            self.f_e = 0
        else:
            self.f_e = self.K_e * ( u - self.x_e )
        
        self.x_o_dot = (self.x_o - self.x_o_prev) / self.dt
        self.x_o_prev = self.x_o

        # Calculate the error given by "f_d" - "f_e"
        self.observation = self.f_e

        info = {
            'x_o' : self.x_o,
            'x_e' : self.x_e,
            'u_h' : u_h,
            'u_r' : u_r_star,
            'u' : u,
            'f_e' : self.f_e,
        }

        return self.observation, info