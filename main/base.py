from numpy import ndarray
import numpy as np
from typing import List
from copy import deepcopy

from .layers import Layer
from .losses import Loss
from .optimizers import Optimizer
from ..utils.helpers import permute_data

class NeuralNetwork(object):
    def __init__(self, layers: List[Layer], loss: Loss, seed: float = 1):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, 'seed', self.seed)

    def forward(self, x_batch: ndarray) -> ndarray:
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)

        return x_out

    def backward(self, loss_gradient: ndarray) -> None:
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return None

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())

        return loss

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_gradients(self):
        for layer in self.layers:
            yield from layer.param_gradients


class Trainer(object):
    def __init__(self, net: NeuralNetwork, optimizer: Optimizer):
        self.net = net
        self.optimizer = optimizer
        self.best_loss = 1e9
        setattr(self.optimizer, 'net', self.net)

    def generate_batches(self,
                         X: ndarray,
                         y: ndarray,
                         size: int = 32):
        assert X.shape[0] == y.shape[0]
        N = X.shape[0]
        for i in range(0, N, size):
            X_batch, y_batch = X[i:i + size], y[i:i + size]
            yield X_batch, y_batch

    def fit(self, X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test:ndarray, epochs, eval_every:int=10,
            batch_size:int=32, seed:int=1, restart:bool=True) -> None:
        np.random.seed(seed)

        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9

        last_model = None
        for e in range(epochs):
            if (e + 1) % eval_every == 0:
                last_model = deepcopy(self.net)

            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            for i, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train_batch(X_batch, y_batch)
                self.optimizer.step()

            if (e+1) % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)

                if loss < self.best_loss:
                    print(f"Validation loss after {e + 1} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    print(
                        f"""Loss increased after epoch {e + 1}, final loss was {self.best_loss:.3f}, using the model from epoch {e + 1 - eval_every}""")
                    self.net = last_model
                    setattr(self.optimizer, 'net', self.net)
                    break