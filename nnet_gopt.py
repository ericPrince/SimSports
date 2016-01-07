from __future__ import division
from math import exp
from random import random, getrandbits, randrange

import numpy as np

#neuron in a neural net layer
class Neuron:
    def __init__(self, weights, bias, func):
        self.weights = weights
        self.bias = bias
        self.func = func

    def __call__(self, input):
        return self.func(self.bias + sum( [w * i for (w, i) in zip(self.weights, input)] ))

    def __len__(self):
        return len(self.weights)

# layer in a neural net
class Layer:
    def __init__(self):
        self.neurons = []

    def __call__(self, input):
        return [neuron(input) for neuron in self.neurons]

    def add_neuron(self, neuron):
        self.neurons.append(neuron)

    def __len__(self):
        return len(self.neurons)

# simple neural net
class Net:
    def __init__(self):
        self.layers = []

    def __call__(self, input):
        output = [i for i in input]
        for layer in self.layers:
            output = layer(output)
        return output

    def add_layer(self):
        self.layers.append(Layer())

    # do the inputs/outputs of layers match up?
    def valid(self):
        for n in range(len(layers) - 1):
            if len(layers[n].neurons) != len(layers[n + 1].weights):
                return False
        return True

    def __len__(self):
        return len(self.layers)

# typical activation functions for neural net
class Activation_Funcs:
    @staticmethod
    def tansig(x):
        return 2 / (1 + exp(-2 * x)) - 1

    @staticmethod
    def logsig(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def lin(x):
        return x

    @staticmethod
    def linstep(x):
        return max(0, min(x, 1))

    @staticmethod
    def step(x):
        return (x >= 0 and 1.0) or 0.0

# genetic optimizer function...
def genetic_optimizer(mins, maxes, fitness_func, callback=None, res=8, pop_size=8, \
                      num_mut=1, num_elite=1, num_gen=50, initpop=None):
    # genes per chromosome
    num_genes = len(mins)

    def get_gene(chrm, m):
        raw = (chrm >> (res * m)) & (2**res - 1)
        return mins[m] + (maxes[m] - mins[m]) / 2**res * float(raw)

    def get_parms(chrm):
        return [get_gene(chrm, m) for m in range(num_genes)]

    def flip_bit(chrm, idx):
        return chrm ^ (1 << idx)

    def crossover(chrm1, chrm2, idx):
        c1 = (chrm1 & ((~0L) << idx)) | (chrm2 & ~((~0L) << idx))
        c2 = (chrm2 & ((~0L) << idx)) | (chrm1 & ~((~0L) << idx))
        return (c1, c2)

    def random_weight_idx(weights):
        r = random() * sum(weights)
        idx = -1
        while r >= 0:
            idx = idx + 1
            r = r - weights(idx)
        return idx

    # create initial population
    ###pop = np.empty(pop_size, dtype=long)
    pop = []
    for n in range(pop_size):
        ###pop[n] = getrandbits(num_genes * res)
        pop.append(getrandbits(num_genes * res))

    # add initial population members to pop
    if initpop:
        for n in range(len(initpop)):
            pop[n] = initpop[n]

    # optimize
    for generation in range(num_gen):
        # evaluate generation
        all_parms = [get_parms(chrm) for chrm in pop]
        fitnesses = np.array(fitness_func(all_parms))
        #fitnesses = np.array([fitness_func(get_parms(chrm)) for chrm in pop])

        # callback
        callback and callback(all_parms, fitnesses)

        # new population and crossover
        ###newpop = np.emptylike(pop)
        newpop = []
        for n in range(pop_size / 2):
            (c1, c2) = crossover(pop[random_weight_idx(fitnesses)], \
                                 pop[random_weight_idx(fitnesses)])
            newpop.append(c1)
            newpop.append(c2)
            #newpop[2*n]     = c1
            #newpop[2*n + 1] = c2

        # retain elites
        for n in range(num_elite):
            idx = fitnesses.argmax()
            newpop[n] = pop[idx]
            pop.delete(idx)

        # random mutations - not for elites
        for n in range(num_elite, pop_size):
            for m in range(num_mut):
                newpop[n] = flip_bit(newpop[n], randrange(num_genes * res))

        pop = newpop

    # return best from last generation (throws out very 
    # final crossover/mutation b/c fitnesses are not evaluated)
    return get_parms(pop[0])