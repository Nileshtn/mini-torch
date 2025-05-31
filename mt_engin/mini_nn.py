import random
from mt_engin.mini_tensor import Tensor

class Neuron:
    def __init__(self, no_of_input):
        self.weight = [Tensor(random.random()) for _ in range(no_of_input)]
        self.bias = Tensor(0)

    def __call__(self, data):
        if len(data) != len(self.weight):
            raise ValueError("Input size mismatch")
        out = sum((w * x for w, x in zip(self.weight, data)), self.bias)
        return out.relu()

    def parameters(self):
        return self.weight + [self.bias]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

class Linear:
    def __init__(self, input_size, output_size):
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()
