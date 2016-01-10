# gopt.py
from __future__ import division
from math import exp
from random import random, getrandbits, randrange

import numpy as np

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
            r = r - weights[idx]
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
        callback and callback(generation, all_parms, fitnesses)

        # new population and crossover
        ###newpop = np.emptylike(pop)
        newpop = []
        for n in range(int(pop_size / 2)):
            rand_idx = randrange(2**res * num_genes)
            (c1, c2) = crossover(pop[random_weight_idx(fitnesses)], \
                                 pop[random_weight_idx(fitnesses)], \
                                 rand_idx)
            newpop.append(c1)
            newpop.append(c2)
            #newpop[2*n]     = c1
            #newpop[2*n + 1] = c2

        # retain elites
        for n in range(num_elite):
            idx = fitnesses.argmax()
            newpop[n] = pop[idx]
            #pop.delete(idx)
            del pop[idx]

        # random mutations - not for elites
        for n in range(num_elite, pop_size):
            for m in range(num_mut):
                newpop[n] = flip_bit(newpop[n], randrange(num_genes * res))

        pop = newpop

    # return best from last generation (throws out very 
    # final crossover/mutation b/c fitnesses are not evaluated)
    return get_parms(pop[0])

def main():
    mins  = [-1, -1, -1]
    maxes = [1, 1, 1]

    def fitness_func(all_parms):
        return [exp(-sum(p)*sum(p)) for p in all_parms]

    def callback(gen_num, all_parms, fitnesses):
        print '======'
        print 'gen # : ' + str(gen_num)

        for (parms, fitness) in zip(all_parms, fitnesses):
            print 'parms   : ' + str(parms)
            print 'fitness : ' + str(fitness)

        print '------'
        idx = fitnesses.argmax()
        best     = all_parms[idx]
        best_fit = fitnesses[idx]
        print 'best parms   : ' + str(best)
        print 'best fitness : ' + str(fitness_func([best,])[0])

        print '======'

    best = genetic_optimizer(mins, maxes, fitness_func, callback)

    print 'done!'
    print 'best parms   : ' + str(best)
    print 'best fitness : ' + str(fitness_func([best,])[0])
    print '--------------'
    print 'globally...'
    global_best = [0, 0, 0]
    print 'best parms   : ' + str(global_best)
    print 'best fitness : ' + str(fitness_func([global_best,])[0])

if __name__ == '__main__':
    main()