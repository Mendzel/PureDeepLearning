import numpy as np
from numpy import ndarray

from PureDeepLearning.main.base import NeuralNetwork


def mae(y_true: ndarray, y_pred: ndarray):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: ndarray, y_pred: ndarray):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model: NeuralNetwork,
                          X_test: ndarray,
                          y_test: ndarray):
    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))
    print('#####################################')