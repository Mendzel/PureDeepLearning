import numpy as np
from numpy import ndarray

def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def to_2d_np(a: ndarray,
             type: str = "col") -> ndarray:
    assert a.ndim == 1, \
        "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    else:
        return a.reshape(1, -1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.array([el if el > 0 else 0 for el in x])

def softmax(x, axis: int=0):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)