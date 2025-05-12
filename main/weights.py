from numpy import ndarray
import numpy as np

from PureDeepLearning.main.actions import ParametersAction

class WeightMultiply(ParametersAction):
    def __init__(self, W: ndarray):
        super().__init__(W)

    def get_output(self) -> ndarray:
        return self.X @ self.param

    def get_input_gradient(self, y_gradient: ndarray) -> ndarray:
        return y_gradient @ np.transpose(self.param, (1,0))

    def get_param_gradient(self, y_gradient: ndarray) -> ndarray:
        return np.transpose(self.X, (1,0)) @ y_gradient

class BiasAdd(ParametersAction):
    def __init__(self, B: ndarray):
        assert B.shape[0] == 1
        super().__init__(B)

    def get_output(self) -> ndarray:
        return self.X + self.param

    def get_input_gradient(self, y_gradient: ndarray) -> ndarray:
        return np.ones_like(self.X) * y_gradient

    def get_param_gradient(self, y_gradient: ndarray) -> ndarray:
        bias_gradient = np.ones_like(self.param) * y_gradient
        return np.sum(bias_gradient, axis=0).reshape(1, bias_gradient.shape[1])