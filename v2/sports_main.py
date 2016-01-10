# sports_main.py
from __future__ import print_function

import numpy as np

import sport
import sim_sports
import nnet
import gopt

def main():
    num_players = 3
    num_teams   = 8

    game_parms = sport.Game_Parms()

    # reasonable value for nnet weight max
    # is order of magnitude similar to max
    # possible acceleration
    w_max = 10.0 * game_parms.a
    #w_max = 1.0 / game_parms.a

    sample_nnet = nnet_layout(num_players)
    num_parms = sum([len(layer)*(len(layer[0])+1) for layer in sample_nnet.layers])

    mins  = -w_max * np.ones(num_parms)
    maxes =  w_max * np.ones(num_parms)

    # get fitness function for optimizer
    f_fitness = generate_fitness_func(num_players, game_parms)

    # assign callback which is called each generation
    output_fn = 'optout.txt'
    callback = callback_save_scores(output_fn)

    # run the genetic optimizer
    best_parms = gopt.genetic_optimizer(mins, maxes, f_fitness, callback, \
                                        pop_size=num_teams, \
                                        num_mut=1, \
                                        num_elite=1, \
                                        num_gen=500, \
                                        initpop=None)

    print(best_parms)
    print('done main')

#-----------------------------
# helper funcs

def callback_save_scores(fn):
    def callback(gen_num, all_parms, fitnesses):
        with open(fn, 'a') as f:
            print('gen: '+str(gen_num), file=f)
            print('fit: '+str(fitnesses), file=f)
            print('------------', file=f)

    return callback

# create a fitness func that is compatible with an optimizer
def generate_fitness_func(n, game_parms):
    def fitness_func(all_parms):
        # create team list from list of all parms
        teams = []
        for parms in all_parms:
            strat = parms2strategy(parms, n)
            teams.append(sport.Team(strat, n))

        # play a season
        season_rankings = sim_sports.play_season(game_parms, teams)

        return season_rankings

    return fitness_func

# create a strategy function from parms
def parms2strategy(parms, n):
    net = parms2nnet(parms, n)

    def strat(input):
        output = net(input)
        ax = np.empty(n)
        ay = np.empty(n)

        for i in range(n):
            ax[i] = output[i]
            ay[i] = output[n + i]

        return ax, ay

    return strat

def parms2nnet(parms, n):
    # generator for getting next parm
    def _gen():
        i = 0
        while True:
            yield parms[i]
            i = i + 1
    p = _gen()

    # create nnet layout with all 0 parameters
    net = nnet_layout(n)

    # populate net with actual parameters
    for layer in net.layers:
        # set layer biases
        for neuron in layer.neurons:
            neuron.b = p.next()
        # set layer weights
        for neuron in layer.neurons:
            for i in range(len(neuron)):
                neuron.w[i] = p.next()

    return net

def nnet_layout(n):
    layer_sizes = [8*n, 8*n, 2*n]
    funcs = [nnet.logsig, nnet.lin]

    return nnet.Net(layer_sizes, funcs)

if __name__ == '__main__':
    main()