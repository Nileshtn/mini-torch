import random  # Import the random module for initializing weights randomly
from mt_engin.mini_tensor import Tensor  # Import the custom Tensor class

class Neuron:
    def __init__(self, no_of_input):
        # Initialize a neuron with a list of random weights and a bias
        self.weight = [Tensor(random.random()) for _ in range(no_of_input)]
        self.bias = Tensor(0)

    def __call__(self, data):
        # Forward pass: computes weighted sum + bias, then applies ReLU
        if len(data) != len(self.weight):
            raise ValueError("Input size mismatch")
        out = sum((w * x for w, x in zip(self.weight, data)), self.bias)
        return out.relu()

    def parameters(self):
        # Returns all parameters (weights and bias) as a list
        return self.weight + [self.bias]

    def zero_grad(self):
        # Sets gradients of all parameters to zero
        for p in self.parameters():
            p.grad = 0

class Linear:
    def __init__(self, input_size, output_size):
        # A fully connected layer with output_size neurons, each with input_size inputs
        self.input_size, self.output_size = input_size, output_size
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def __call__(self, x):
        # Forward pass: applies each neuron to the input
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Returns all parameters of all neurons in this layer
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        # Sets gradients of all parameters in all neurons to zero
        for n in self.neurons:
            n.zero_grad()

class Relu:
    def __call__(self, x:list[Tensor]):
        # Applies ReLU activation to each Tensor in the input list
        out = []
        for value in x:
            out.append(value.relu())
        return out

class Sigmoid:
    def __call__(self, x:list[Tensor]):
        # Applies Sigmoid activation to each Tensor in the input list
        out = []
        for value in x:
            out.append(value.sigmoid())
        return out

class Module:
    def __init__(self, *args, **kwargs):
        # Base class for all neural network modules
        pass
    
    def parameters(self):
        # Collects parameters from all attributes that are Neuron or Linear
        parametes = []
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, (Neuron, Linear)):
                parametes.extend(attr_val.parameters())
        return parametes
    
    def zero_grad(self):
        # Sets gradients to zero for all attributes that are Neuron or Linear
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, (Neuron, Linear)):
                attr_val.zero_grad()
            
    def forward(self, *args, **kwargs):
        # To be implemented by subclasses: defines the forward pass
        raise NotImplementedError("Subclasses must override process()")
    
    def __repr__(self):
        # Returns a string representation of the module and its layers
        output = []
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, (Linear)):
                output.append(f"{attr_val.__class__.__name__} -> {attr_val}({attr_val.input_size, attr_val.output_size})")
                
    def __call__(self, *args, **kwargs):
        # Makes the module callable, calls forward()
        return self.forward(*args, **kwargs)