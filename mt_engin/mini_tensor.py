import math  # Import math module for mathematical functions

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        # Initialize a Tensor object.
        # data: the scalar value of the tensor
        # _children: parent tensors (for autograd graph)
        # _op: operation that produced this tensor (for graph visualization/debugging)
        self.data = data
        self.grad = 0  # Gradient of this tensor (used in backprop)
        self._backward = lambda: None  # Function to compute the gradient for this tensor
        self._prev = set(_children)  # Set of parent tensors
        self._op = _op  # Operation that produced this tensor

    def __add__(self, other):
        # Overload the + operator for tensors
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad  # d(out)/d(self) = 1
            other.grad += out.grad  # d(out)/d(other) = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        # Overload the * operator for tensors
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad  # d(out)/d(self) = other.data
            other.grad += self.data * out.grad  # d(out)/d(other) = self.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        # Overload the ** operator for tensors (power)
        assert isinstance(other, (int, float)), "Only int/float exponents supported"
        out = Tensor(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad  # Power rule
        out._backward = _backward
        return out

    def relu(self):
        # ReLU activation: max(0, x)
        out = Tensor(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad  # Gradient is 1 if out > 0 else 0
        out._backward = _backward
        return out
    
    def sigmoid(self):
        # Sigmoid activation: 1 / (1 + exp(-x))
        out = Tensor(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid')
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad  # Derivative of sigmoid
        out._backward = _backward
        return out

    def backward(self):
        # Backpropagation: compute gradients for all tensors in the computation graph
        topo = []  # List for topological order
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1  # Seed gradient (usually for loss)
        for v in reversed(topo):
            v._backward()  # Call backward for each tensor in reverse topological order

    # Operator overloads for convenience
    def __neg__(self): return self * -1  # -self
    def __radd__(self, other): return self + other  # other + self
    def __sub__(self, other): return self + (-other)  # self - other
    def __rsub__(self, other): return other + (-self)  # other - self
    def __rmul__(self, other): return self * other  # other * self
    def __truediv__(self, other): return self * other**-1  # self / other
    def __rtruediv__(self, other): return other * self**-1  # other / self

    def __repr__(self):
        # String representation for debugging
        return f"Value(data={self.data}, grad={self.grad})"