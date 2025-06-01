import random  # For generating random data
from mt_engin.mini_tensor import Tensor  # Import the custom Tensor class
from mt_engin.mini_nn import Module  # Import the base Module class for models


class MiniOptimizer:
    def __init__(self, model:Module, lr=0.0001):
        # Initialize the optimizer with a model and learning rate
        self.model = model  # The model whose parameters will be updated
        self.lr = lr  # Learning rate

    def zero_grad(self):
        # Set all gradients in the model to zero
        self.model.zero_grad()

    def step(self):
        # Update each parameter in the model using gradient descent
        for p in self.model.parameters():
            p.data += -self.lr * p.grad  # Gradient descent update rule

def mse_loss(pred:Tensor, target:Tensor):
    # Mean squared error loss for two scalar tensors
    return (pred - target) ** 2

def target_generator():
    # Generate a random binary input vector and a target label
    x = [random.randint(0, 1) for _ in range(5)]  # Random binary vector of length 5
    y = 0 if x.count(1) < 3 else 1  # Target is 1 if at least 3 ones, else 0
    x_t = [Tensor(val) for val in x]  # Convert input to Tensor objects
    return x_t, Tensor(y)  # Return input and target as Tensors

def binary_round(val, threshold=0.5):
    # Convert a Tensor value to 0 or 1 based on a threshold
    return Tensor(1) if val.data >= threshold else Tensor(0)

# def save(model):
# Placeholder for a model saving function (not implemented)


