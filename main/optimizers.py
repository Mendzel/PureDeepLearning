import numpy as np


class Optimizer(object):
    def __init__(self, lr: float = 0.01, final_lr: float = 0, decay_type: str = 'exponential'):
        self.decay_per_epoch = None
        self.max_epochs = None
        self.lr = lr
        self.final_lr = final_lr
        self.decay_type = decay_type

    def step(self) -> None:
        pass

    def setup_decay(self) -> None:
        if not self.decay_type:
            return
        elif self.decay_type == 'exponential':
            self.decay_per_epoch = np.power(self.final_lr / self.lr, 1.0 / (self.max_epochs - 1))
        elif self.decay_type == 'linear':
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)

    def decay_lr(self) -> None:
        if not self.decay_type:
            return
        elif self.decay_type == 'exponential':
            self.lr *= self.decay_per_epoch
        elif self.decay_type == 'linear':
            self.lr -= self.decay_per_epoch

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self):
        for (param, param_gradient) in zip(self.net.params(), self.net.param_gradients()):
            param -= self.lr * param_gradient

class SGDMomentum(Optimizer):
    def __init__(self, lr: float=0.01, momentum: float=0.9, final_lr: float = 0, decay_type: str = None):
        super().__init__(lr, final_lr, decay_type)
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