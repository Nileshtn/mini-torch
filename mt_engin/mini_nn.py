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
        self.input_size, self.output_size = input_size, output_size
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

class Relu:
    def __call__(self, x:list[Tensor]):
        out = []
        for value in x:
            out.append(value.relu())

        return out

class Sigmoid:
    def __call__(self, x:list[Tensor]):
        out = []
        for value in x:
            out.append(value.sigmoid())
        return out

class Module:
    def __init__(self, *args, **kwargs):
        pass
    
    def parameters(self):
        parametes = []
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, (Neuron, Linear)):
                parametes.extend(attr_val.parameters())
        return parametes
    
    def zero_grad(self):
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, (Neuron, Linear)):
                attr_val.zero_grad()
            
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must override process()")
    
    def __repr__(self):
        output = []
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, (Linear)):
                output.append(f"{attr_val.__class__.__name__} -> {attr_val}({attr_val.input_size, attr_val.output_size})")
                
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)