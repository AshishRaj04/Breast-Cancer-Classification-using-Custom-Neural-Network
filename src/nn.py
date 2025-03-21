import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        if not isinstance(nin, int) or nin <= 0:
            raise ValueError("nin must be a positive integer")
        self.w = [Value(random.random()) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"Expected input of size {len(self.w)}, but got {len(x)}")
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = z.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        if (
            not isinstance(nin, int)
            or nin <= 0
            or not isinstance(nout, int)
            or nout <= 0
        ):
            raise ValueError("nin and nout must be positive integers")
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params


class MLP:
    def __init__(self, nin, nouts):
        if not isinstance(nin, int) or nin <= 0:
            raise ValueError("nin must be a positive integer")
        if not isinstance(nouts, (list, tuple)) or not all(
            isinstance(n, int) and n > 0 for n in nouts
        ):
            raise ValueError("nouts must be a list/tuple of positive integers")
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params
