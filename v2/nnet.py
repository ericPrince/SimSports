# nnet.py
from math import exp
import numpy as np

class Net:
    # create a Net with layers given as a list in
    # layer_sizes, and activation functions given as
    # a list in funcs
    def __init__(self, layer_sizes, funcs):
        self.layers = []
        
        for i in range(1, len(layer_sizes)):
            self.layers.append(           \
                Layer(layer_sizes[i - 1], \
                      layer_sizes[  i  ], \
                      funcs[i - 1]        ))

    # return the net's output for the given input arraylike
    def __call__(self, input):
        out = np.array(input)

        for layer in self.layers:
            out = layer(out)

        return out

    def __getitem__(self, layer):
        return self.layers[layer]

    # return number of layers in the net
    def __len__(self):
        return len(self.layers)

    # add an empty layer to the net??
    def append(self):
        self.layers.append(Layer())

    # do number of inputs/outputs of layers match??
    def valid(self):
        pass

class Layer:
    def __init__(self):
        self.neurons = []

    def __init__(self, numInputs, numOutputs, func):
        self.neurons = []

        for i in range(numOutputs):
            self.neurons.append(Neuron(func=func, numInputs=numInputs))

    def __call__(self, input):
        return np.array([n(input) for n in self.neurons])

    def __len__(self):
        return len(self.neurons)

    def __getitem__(self, n):
        return self.neurons[n]

class Neuron:
    def __init__(self, func, weights=None, bias=None, numInputs=None):
        self.w = None
        if weights is not None:
            self.w = np.array(weights)
        elif numInputs:
            self.w = np.zeros(numInputs)

        self.b = bias or 0
        self.f = func

    def __call__(self, input):
        return self.f(self.b + self.w.dot(input))

    def __len__(self):
        return self.w.size

def logsig(x):
    if x > 1e2:
        return 1.0
    elif x < -1e2:
        return -1.0
    return 1 / (1 + exp(-x))

def tansig(x):
    if x > 1e2:
        return 1.0
    elif x < -1e2:
        return -1.0
    return 2 / (1 + exp(-2 * x)) - 1

def lin(x):
    return x

def linstep(x):
    return max(0.0, min(x, 1.0))

def step(x):
    return (x >= 0 and 1.0) or 0.0


def main():
    layer_sizes = [4, 3, 2, 1]
    funcs = [logsig, tansig, lin, step]

    net = Net(layer_sizes, funcs)

if __name__ == '__main__':
    main()