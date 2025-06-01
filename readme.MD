# 🧠 Mini-Torch
##### Inspired by Andrej Karpathy's micrograd
---
**mini-torch** is a lightweight neural network framework and autograd engine built from scratch using pure Python. It’s designed to demystify how deep learning models work under the hood—no NumPy, no PyTorch, just raw Python.

---

## 🚀 Features

- `Tensor` class with:
  - Backpropagation via autograd
  - Operator overloading (`+`, `-`, `*`, `/`, `**`, etc.)
  - ReLU activation
- Simple fully connected layers (`Neuron`, `Linear`)
- SGD optimizer (`MiniOptimizer`)
- Mean Squared Error (MSE) loss
- Toy binary classification dataset generator

---

## 🛠️ Example Usage

```python
from layers import Linear
from training_utils import MiniOptimizer, mse_loss, target_generator

model = Linear(5, 1)
optimizer = MiniOptimizer([model], lr=0.1)

# Training loop
for epoch in range(100):
    x, y = target_generator()
    pred = model(x)
    loss = mse_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.data}")

```

# Model Training Setup

This markdown documents the setup and training of a neural model using the `Tensor`, `Neuron`, and `Linear` classes.

---

### Problem Description

The task is to predict the most frequent value in a list of 5 binary inputs (each either 0 or 1). The model is trained to output the dominant value in the list.

---

### Input and Target

* **Input**: List of 5 `Tensor` objects, each with a value of `0` or `1`.
* **Target**: A scalar `Tensor`, which is either `0` or `1`, representing the value (0 or 1) that appears most frequently in the input list.

Example:

```python
x = [Tensor(0), Tensor(1), Tensor(1), Tensor(1), Tensor(0)]
# Most common value is 1
# So, target = Tensor(1)
```

---

### Model Architecture: Deep Feedforward Neural Network

This section describes a deeper network defined using custom modules:

```python
class NuralNetwork(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.l1 = nn.Linear(in_channel, 4)
        self.l2 = nn.Linear(4, 8)
        self.l3 = nn.Linear(8, 4)
        self.l4 = nn.Linear(4, out_channel)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x
```

* 4-layer fully connected neural network
* No non-linearity used between layers in this version (can be improved by adding ReLU/Sigmoid)
* Input: 5-dimensional binary Tensor list
* Output: Single scalar Tensor (predicted dominant class)

---

### Loss and Optimizer

```python
optimizer = MiniOptimizer(model, 0.000001)
```

* Optimizer performs basic SGD using `.backward()` and `.step()`
* Loss function used is MSE (mean squared error):

```python
mse_loss(out, y)
```

---

### Dataset and Target Generator

```python
from mt_engin.utils import target_generator
```

* `target_generator()` creates a 5-element input and its corresponding majority target

---

### Training Loop

```python
epoch = 500
batch = 32
for epoch in range(epoch):
    total_loss = Tensor(0)
    for _ in range(batch):
        x, y = target_generator()
        out = model(x)
        total_loss += mse_loss(out, y)

    optimizer.zero_grad()
    avg_loss = total_loss / 32
    avg_loss.backward()
    optimizer.step()
    print("loss:", avg_loss)
```

* For each epoch:

  * Accumulates loss over 32 samples
  * Computes average loss and backpropagates
  * Optimizer updates weights using computed gradients

---

### Example Output

```
epoch 0: loss: Value(data=0.2498, grad=0)
epoch 1: loss: Value(data=0.2471, grad=0)
...
epoch 499: loss: Value(data=0.0012, grad=0)
```

This shows loss decreasing over time, indicating learning progress.

---

### Evaluation Example

After training, you can evaluate the model on individual examples using the following code:

```python
x, y = target_generator()
out = model(x)
out = binary_round(out)
for i in x:
    print(i.data, end=" ")
print("target", y.data)
print("pred", out.data)
```

This will print the input values, the actual target, and the predicted output after rounding.

---



# Autograd Engine with Tensor Class

This markdown file documents the implementation of a minimal autograd engine using a custom `Tensor` class in Python.

---

## Overview

This engine supports reverse-mode automatic differentiation (autodiff) for scalar tensors. It includes basic arithmetic operations, ReLU and sigmoid activations, and gradient backpropagation.

---

## Class: `Tensor`

```python
class Tensor:

A scalar-based tensor that supports autodiff.

### Constructor

```python
def __init__(self, data, _children=(), _op=''):
```

* `data`: Scalar value
* `_children`: Tuple of parent Tensors
* `_op`: String denoting operation that created this tensor

---

## Operations

### Addition

```python
def __add__(self, other):
```

* Adds two tensors
* Gradient: $\frac{d}{dx}(x + y) = 1$

### Multiplication

```python
def __mul__(self, other):
```

* Multiplies two tensors
* Gradient via product rule:
  $\frac{d}{dx}(x \cdot y) = y \quad \text{and} \quad \frac{d}{dy}(x \cdot y) = x$

### Power

```python
def __pow__(self, other):
```

* Raises tensor to scalar power
* Gradient: $nx^{n-1}$

---

## Activation Functions

### ReLU

```python
def relu(self):
```

* $\text{ReLU}(x) = \max(0, x)$
* Gradient: 1 if $x > 0$, else 0

### Sigmoid

```python
def sigmoid(self):
```

* $\sigma(x) = \frac{1}{1 + e^{-x}}$
* Gradient: $\sigma(x)(1 - \sigma(x))$

---

## Backward Pass

### `.backward()`

```python
def backward(self):
```

* Computes gradients using reverse-mode autodiff
* Builds a topological graph, initializes `self.grad = 1`, and propagates gradients in reverse order

---

## Operator Overloads

Support for syntactic sugar:

```python
def __neg__(self): return self * -1

def __radd__(self, other): return self + other

def __sub__(self, other): return self + (-other)

def __rsub__(self, other): return other + (-self)

def __rmul__(self, other): return self * other

def __truediv__(self, other): return self * other**-1

def __rtruediv__(self, other): return other * self**-1
```

---

## Representation

```python
def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
```

Prints tensor value and gradient in a readable format.

---

## Computation Graph (Conceptual)

Each operation builds a computation graph. Example:

```python
x = Tensor(2.0)
y = Tensor(3.0)
z = x * y + y
z.backward()
```

Graph:

```
   x       y
    \     /
     *   y
      \ /
       +
       |
       z
```

Backpropagation will compute:

* $\frac{dz}{dx} = y = 3.0$
* $\frac{dz}{dy} = x + 1 = 2.0 + 1.0 = 3.0$

---

## Neural Modules

These classes create basic neural network components using the `Tensor` class with autograd support.

---

### Class: `Neuron`

A single neuron with ReLU activation.

```python
class Neuron:
```

* **Constructor**:
  `__init__(no_of_input)`
  Initializes a list of weights (as `Tensor`) and a bias term.

* **Forward call**:
  `__call__(data)`
  Applies a weighted sum of the inputs and a bias, then applies ReLU:
  $\text{output} = \text{ReLU}(\sum_i w_i x_i + b)$

* **parameters()**:
  Returns the list of all trainable `Tensor` objects (weights + bias).

* **zero\_grad()**:
  Resets gradients of all parameters to zero.

---

### Class: `Linear`

A fully connected layer composed of multiple neurons.

```python
class Linear:
```

* **Constructor**:
  `__init__(input_size, output_size)`
  Creates `output_size` neurons, each expecting `input_size` inputs.

* **Forward call**:
  `__call__(x)`
  Passes the input to each neuron and collects the outputs.

* **parameters()**:
  Returns a flat list of all trainable parameters in all neurons.

* **zero\_grad()**:
  Clears gradients in all contained neurons.

---

### Class: `Relu`

Applies ReLU activation elementwise.

```python
class Relu:
```

* ****call**(x: list\[Tensor])**:
  Applies `relu()` on each `Tensor` in the input list.

---

### Class: `Sigmoid`

Applies Sigmoid activation elementwise.

```python
class Sigmoid:
```

* ****call**(x: list\[Tensor])**:
  Applies `sigmoid()` on each `Tensor` in the input list.

---

### Class: `Module`

A base class for building neural network models.

```python
class Module:
```

* **parameters()**:
  Recursively gathers all trainable parameters from `Neuron` and `Linear` attributes.

* **zero\_grad()**:
  Recursively resets all gradients in submodules.

* **forward(...)**:
  Meant to be overridden by subclasses. Defines the forward pass logic.

* ****call**(...)**:
  Calls `forward(...)`.

* ****repr**()**:
  Summarizes the model architecture. Currently partially implemented.

