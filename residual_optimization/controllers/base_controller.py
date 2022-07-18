import numpy as np

class BaseController():
    """
    This class serves as a super for a simple controller.
    """
    def __init__(
        self,
        observation_shape=(1,),
        observation_min=-1,
        observation_max=1,
        action_shape=(1,),
        action_min=-1,
        action_max=1
    ):
        self.observation_shape = observation_shape
        self.observation_min = observation_min
        self.observation_max = observation_max
        self.action_shape = action_shape
        self.action_min = action_min
        self.action_max = action_max
        self.reference = np.zeros(shape=self.observation_shape)

    def update(self, observation):
        raise NotImplementedError

    def set_reference(self, reference):
        self.reference = reference