from numpy import ndarray

from PureDeepLearning.utils.assertions import assert_same_shape


class Action(object):  # Moze bedzie lepiej nazwa Operation
    def __init__(self):
        self.X_gradient = None
        self.y = None
        self.X = None

    def forward(self, X: ndarray, inference: bool = False):
        self.X = X
        self.y = self.get_output(inference=inference)
        return self.y

    def backward(self, y_gradient: ndarray) -> ndarray:
        assert_same_shape(self.y, y_gradient)
        self.X_gradient = self.get_input_gradient(y_gradient)
        assert_same_shape(self.X, self.X_gradient)
        return self.X_gradient

    def get_output(self, **kwargs) -> ndarray:
        raise NotImplementedError

    def get_input_gradient(self, y_gradient: ndarray) -> ndarray:
        raise NotImplementedError


class ParametersAction(Action):
    def __init__(self, param: ndarray):
        super().__init__()
        self.param_gradient = None
        self.param = param

    def backward(self, y_gradient: ndarray) -> ndarray:
        assert_same_shape(self.y, y_gradient)
        self.X_gradient = self.get_input_gradient(y_gradient)
        self.param_gradient = self.get_param_gradient(y_gradient)
        assert_same_shape(self.X, self.X_gradient)
        assert_same_shape(self.param, self.param_gradient)
        return self.X_gradient

    def get_param_gradient(self, output_gradient: ndarray) -> ndarray:
        raise NotImplementedError
