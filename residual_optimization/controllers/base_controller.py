import numpy as np

class BaseController():
    """
    This class serves as a super for a simple controller.
    """
    def __init__(
        self,
        feedback_shape : tuple = (1,),
        control_shape : tuple = (1,),
        reference_shape : tuple = (1,),
    ):
        """
        Reference and observation should have the same shape. The control objective is 
        to bring the observations as close to reference as possible.

        Parameters
        ----------
        observation_shape : tuple
            The shape of the feedback of the controller.
        control_shape : tuple
            The shape of the controlled variable of the controller.
        reference_shape : tuple
            The shape of the reference variable.
        """
        self.feedback_shape = feedback_shape
        self.control_shape = control_shape
        self.reference_shape = reference_shape
        self.time = 0

    def reset(self):
        """
        Reset all internal variables to starting states.
        """
        self.time = 0

    def set_reference(self, reference : np.ndarray):
        """
        Set the desired reference of the control loop.

        Parameters
        ----------
        reference : np.ndarray
            Set the desired value to be tracked by the control variable.        
        """
        raise NotImplementedError

    def get_reference(self):
        """
        Returns
        ----------
        reference : np.ndarray
            Get the current setpoint of the controller
        """
        raise NotImplementedError

    def update(self, feedback : np.ndarray, dt : np.float64):
        """
        Do one iteration of the control loop.

        Parameters
        ----------
        feedback : np.ndarray
            Feedback variable for the controller.
        dt : np.float64
            The time duration of the control loop.

        Returns
        -------
        control : np.ndarray
            The controlled variable.
        """
        raise NotImplementedError