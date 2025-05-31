# 🧠 mini-torch

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
