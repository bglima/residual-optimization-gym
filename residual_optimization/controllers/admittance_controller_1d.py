import numpy as np

from residual_optimization.controllers.base_controller import BaseController

class AdmittanceController1D(BaseController):
    """
    This class inherits from controllers.BaseController. It represents a 
    unidimensional position controller with a force-feedback loop. 
    Assume that every variable is within end-effector frame, a.k.a.
    a detached end-effector with a moving reference frame.
    
    Refer to the report for further information.

    Superclass variables
    --------------------
    feedback_shape : (1,)
        Only the force is fed back.
    control_shape : (1,)
        A position reference is sent to the robot.
    reference_shape : (2, 1)
        A position and a force reference are given to the robot.

    Internal variables
    ------------------
    x_d : np.ndarray
        1d-array with the reference position
    f_d : np.ndarray
        1d-array with the reference force
    x_c : np.ndarray
        1d-array with the controled position variable
    f_e : np.ndarray
        1d-array with the force exerted by the robot over the environment
    """
    def __init__(self, M_d : np.matrix, K_P : np.matrix, K_D : np.matrix):
        """
        Parameters
        ----------
        M_d : np.matrix
            1d-array with the Intertia matrix
        K_P : np.matrix
            1d-array with the spring constant
        K_D : np.matrix
            1d-array with the damping constant    
        """
        self.M_d = M_d
        self.M_d_inv = M_d ** -1
        self.K_P = K_P
        self.K_D = K_D
        super(AdmittanceController1D, self).__init__(
            feedback_shape = (1,),
            control_shape = (1,),
            reference_shape = (2, 1)
        )
        self.f_d = np.zeros(self.feedback_shape)
        self.f_e = np.zeros(self.feedback_shape)
        self.x_d = np.zeros(self.control_shape)
        self.x_c = np.zeros(self.control_shape)
        self.x_dot_c = np.zeros(self.control_shape)
        self.x_dot_dot_c = np.zeros(self.control_shape)

    def set_reference(self, reference: np.ndarray):
        """
        Set a position followed by a force reference.

        Parameters
        ----------
        reference : np.ndarray
            2d-array of form [x_d, f_d]
        """
        (self.x_d, self.f_d) = reference

    def update(self, feedback : np.ndarray, dt : np.float32):
        """
        Receives a feedback comprised by the force felt by a force sensor.

        Parameters
        ----------
        feedback : np.ndarray
            1d-array with the force read by a force sensor
        dt : np.float32
            Current duration of control loop in seconds.

        Returns
        -------
        x_c : np.ndarray
            1d-array with the controlled variable
        """
        # The force exerted by the robot is the opposite of the force read by the sensor
        self.f_e = -feedback

        # Force error
        delta_f = self.f_d - self.f_e

        # Internal states
        self.x_dot_dot_c = self.M_d_inv * (
            delta_f - \
            self.K_P * (self.x_c - self.x_d) - \
            self.K_D * self.x_dot_c \
        )
        self.x_dot_c += self.x_dot_dot_c * dt
        self.x_c += self.x_dot_c * dt
        
        # Returns the controlled variable
        return self.x_c
