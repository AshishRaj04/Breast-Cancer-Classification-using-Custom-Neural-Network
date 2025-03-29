import math


class Value:
    def __init__(self, data, _children=(), _op="", _label=""):
        self.data = data
        self._backward = lambda: None
        self._prev = set(_children)
        self.grad = 0
        self.op = _op
        self.label = _label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.grad * other.data
            other.grad += self.grad * self.data

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")

        def _backward():
            self.grad += 1.0 / other.data * out.grad
            other.grad += -self.data / other.data**2 * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def tanh(self):
        x = self.data
        e_2x = math.exp(2 * x)
        t = (e_2x - 1) / (e_2x + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        r = max(0, x)
        out = Value(r, (self,), "relu")

        def _backward():
            self.grad += (1 if x > 0 else 0) * out.grad

        out._backward = _backward
        return out

    def pow(self, other):
        assert isinstance(other, (int, float))
        x = self.data
        out = Value(x**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * x ** (other - 1) * out.grad

        out._backward = _backward
        return out
    
    def log(self):
        x = self.data
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1 / x) * out.grad

        out._backward = _backward
        return out
    
    def mean(self):
        out = Value(self.data / len(self._prev) if self._prev else self.data, (self,), "mean")

        def _backward():
            for v in self._prev:
                v.grad += (1 / len(self._prev)) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        stack = []
        visited = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            if hasattr(node, "_prev"):
                for child in node._prev:
                    if isinstance(child, Value):
                        dfs(child)
            stack.append(node)

        dfs(self)
        self.grad = 1.0
        for node in stack[::-1]:
            node._backward()