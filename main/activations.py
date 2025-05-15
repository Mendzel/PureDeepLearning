from numpy import ndarray
import numpy as np

from PureDeepLearning.main.actions import Action

class Sigmoid(Action):
    def __init__(self) -> None:
        super().__init__()

    def get_output(self, **kwargs) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.X))

    def get_input_gradient(self, output_gradient: ndarray) -> ndarray:
        sigmoid_backward = self.y * (1.0 - self.y)
        return sigmoid_backward * output_gradient

class Linear(Action):
    def __init__(self) -> None:
        super().__init__()

    def get_output(self, **kwargs) -> ndarray:
        return self.X

    def get_input_gradient(self, output_gradient: ndarray) -> ndarray:
        return output_gradient