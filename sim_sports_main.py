import numpy as np

import rugby_game
import sim_rugby
import nnet_gopt
import rugby_season

def create_nnet_layout(num_players):
    num_inputs  = 4*num_players + 2
    num_outputs = 2*num_players + 2 + 1

    net = nnet_gopt.Net()

    # 1st layer
    net.add_layer()
    for i in range(num_inputs):
        neuron = nnet_gopt.Neuron(np.zeros(num_inputs), 0, nnet_gopt.Activation_Funcs.logsig)
        net.layers[0].add_neuron(neuron)

    # output layer
    net.add_layer()
    for i in range(num_outputs):
        neuron = nnet_gopt.Neuron(np.zeros(num_inputs), 0, nnet_gopt.Activation_Funcs.linstep)
        net.layers[-1].add_neuron(neuron)

    # pass_flag gets step function
    neuron = nnet_gopt.Neuron(np.zeros(num_inputs), 0, nnet_gopt.Activation_Funcs.step)
    net.layers[-1].add_neuron(neuron)

    return net

def nnet_to_strategy(net, num_players, a_max, v_max):
    def strat(input):
        output = net(input)
        ax = np.empty(num_players)
        ay = np.empty(num_players)

        for i in range(num_players):
            ax[i] = output[i] * a_max
            ay[i] = output[num_players + i] * a_max

        pass_flag = output[2*num_players]
        px = output[2*num_players + 1] * 2*v_max
        py = output[2*num_players + 2] * 2*v_max

        return ax, ay, pass_flag, px, py

    return strat

# also choose activation functions???
def parms_to_nnet(parms, num_players):
    def idx_gen():
        i = 0
        while True:
            yield i
            i = i + 1
    idx = idx_gen()

    net = create_nnet_layout(num_players)

    for layer in net.layers:
        for neuron in layer.neurons:
            neuron.bias = parms[idx.next()]
        for neuron in layer.neurons:
            for i in range(len(neuron)):
                neuron.weights[i] = parms[idx.next()]

    return net

def main():
    # game-play variables
    num_players   = 3
    num_teams     = 8

    a_max         =   2.0
    v_max         =  10.0
    player_radius =   1.0
    fieldx        = 100.0
    fieldy        =  50.0

    time_limit    = 1000.0
    dt            =    1.0

    # neural net / genetic alg variables
    num_inputs  = 4*2*num_players + 4
    num_hidden  =   4*num_players + 2
    num_outputs =   2*num_players + 2 + 1
    num_parms   = (num_inputs + 1) * num_hidden + (num_hidden + 1) * num_outputs

    # set min and max weights to +-1
    mins  = -np.ones(num_parms)
    maxes =  np.ones(num_parms)

    # biases - should this be how they all (weights) are?
    mins[ 0:num_hidden] = -1 - num_inputs
    maxes[0:num_hidden] =  1 + num_inputs

    #biases
    mins[ ((num_inputs + 1) * num_hidden):((num_inputs + 1) * num_hidden) + num_outputs] = -1 - num_hidden
    maxes[((num_inputs + 1) * num_hidden):((num_inputs + 1) * num_hidden) + num_outputs] =  1 + num_hidden

    # create fitness function
    fitness_func = rugby_season.generate_fitness_func(num_players, dt, time_limit, player_radius, fieldx, fieldy, a_max, v_max)

    # run genetic optimizer
    best_parms = nnet_gopt.genetic_optimizer(mins, maxes, fitness_func, pop_size=num_teams, num_gen=100)

    print best_parms
    print 'done main'


if __name__ == '__main__':
    main()