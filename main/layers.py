from typing import List
from numpy import ndarray
import numpy as np

from PureDeepLearning.main.actions import Action, ParametersAction
from PureDeepLearning.main.activations import Sigmoid
from PureDeepLearning.main.weights import WeightMultiply, BiasAdd
from PureDeepLearning.utils.assertions import assert_same_shape


class Layer(object):
    def __init__(self, neurons: int):
        self.X = None
        self.y = None
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.param_gradients: List[ndarray] = []
        self.operations: List[Action] = []

    def create_layer(self, X: ndarray) -> None:
        raise NotImplementedError

    def forward(self, X: ndarray) -> ndarray:
        if self.first:
            self.create_layer(X)
            self.first = False

        self.X = X
        for operation in self.operations:
            X = operation.forward(X)
        self.y = X

        return self.y

    def backward(self, y_gradient: ndarray) -> ndarray:
        assert_same_shape(self.y, y_gradient)

        for operation in reversed(self.operations):
            y_gradient = operation.backward(y_gradient)

        X_gradient = y_gradient
        self.get_param_gradients()

        return X_gradient

    def get_param_gradients(self) -> ndarray:
        self.param_gradients = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParametersAction):
                self.param_gradients.append(operation.param_gradient)

    def get_params(self) -> ndarray:
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParametersAction):
                self.params.append(operation.param)



class Dense(Layer):
    def __init__(self, neurons: int, activation: Action = Sigmoid()):
        super().__init__(neurons)
        self.seed = None
        self.activation = activation

    def create_layer(self, X: ndarray) -> None:
        if self.seed:
            np.random.seed(self.seed)

        self.params = []
        self.params.append(np.random.randn(X.shape[1], self.neurons))
        self.params.append(np.random.randn(1, self.neurons))

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation]
        return None