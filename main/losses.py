from numpy import ndarray
import numpy as np

from PureDeepLearning.utils.assertions import assert_same_shape
from PureDeepLearning.utils.helpers import softmax


class Loss(object):
    def __init__(self):
        self.X_gradient = None
        self.prediction = None
        self.target = None

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target

        loss_value = self.calculate_loss()

        return loss_value

    def backward(self) -> ndarray:
        self.X_gradient = self.get_x_gradient()
        assert_same_shape(self.prediction, self.X_gradient)

        return self.X_gradient

    def calculate_loss(self) -> float:
        raise NotImplementedError

    def get_x_gradient(self) -> ndarray:
        raise NotImplementedError


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def calculate_loss(self) -> float:
        loss = np.sum((self.prediction - self.target) ** 2) / self.prediction.shape[0]

        return loss

    def get_x_gradient(self) -> ndarray:
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, eps: float=1e-9):
        super().__init__()
        self.softmax_predictions = None
        self.eps = eps
        self.single_output = False

    def calculate_loss(self) -> float:
        softmax_predictions = softmax(self.prediction, 1)
        self.softmax_predictions = np.clip(softmax_predictions, self.eps, 1 - self.eps)
        softmax_cross_entropy_loss = (-1.0 * self.target * np.log(self.softmax_predictions) - (1.0 - self.target) * np.log(1 - self.softmax_predictions))
        return np.sum(softmax_cross_entropy_loss)

    def get_x_gradient(self) -> ndarray:
        return self.softmax_predictions - self.target