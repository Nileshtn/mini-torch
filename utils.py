import random
from mini_tensor import Tensor

class MiniOptimizer:
    def __init__(self, layers, lr=0.01):
        self.layers = layers
        self.lr = lr

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def step(self):
        for layer in self.layers:
            for p in layer.parameters():
                p.data += -self.lr * p.grad

def mse_loss(pred, target):
    return (pred - target) ** 2

def target_generator():
    x = [random.randint(0, 1) for _ in range(5)]
    y = 0 if x.count(1) < 3 else 1
    x_t = [Tensor(val) for val in x]
    return x_t, Tensor(y)

def binary_round(val, threshold=0.5):
    return Tensor(1) if val.data >= threshold else Tensor(0)
