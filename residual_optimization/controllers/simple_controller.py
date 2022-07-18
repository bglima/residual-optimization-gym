import numpy as np

from residual_optimization.controllers.base_controller import BaseController

class SimpleController(BaseController):
    """
    This class inherits from BaseController. 
    """
    def __init__(self, *args, **kwargs):
        super(SimpleController, self).__init__(*args, **kwargs)

    def update(self, observation, dt):
        """
        Main control loop, based on the "observation" and on
        the time since last update "dt"
        """
        # TODO: Interaction control logic
        action = np.random.uniform(
            self.action_min,
            self.action_max,
            self.action_shape
        )
        # TODO: Replace temporary return

        return action
