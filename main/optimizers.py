import numpy as np


class Optimizer(object):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self) -> None:
        pass

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self):
        for (param, param_gradient) in zip(self.net.params(), self.net.param_gradients()):
            param -= self.lr * param_gradient

class SGDMomentum(Optimizer):
    def __init__(self, lr: float=0.01, momentum: float=0.9):
        super().__init__(lr)
        self.velocities = None
        self.first = True
        self.momentum = momentum

    def step(self) -> None:
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.params()]
            self.first = False

        for (param, param_gradient, velocity) in zip(self.net.params(), self.net.param_gradients(), self.velocities):
            self.update_rule(param=param, gradient=param_gradient, velocity=velocity)

    def update_rule(self, **kwargs):
        kwargs['velocity'] *= self.momentum
        kwargs['velocity'] += self.lr * kwargs['gradient']
        kwargs['param'] -= kwargs['velocity']