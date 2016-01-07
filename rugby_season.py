import numpy as np

import rugby_game
import sim_rugby
import sim_sports_main

def generate_fitness_func(num_players, dt, time_limit, player_radius, fieldx, fieldy, a_max, v_max):
    def fitness_func(all_parms):
        teams = []
        for parms in all_parms:
            strat = sim_sports_main.nnet_to_strategy(sim_sports_main.parms_to_nnet(parms, num_players), num_players, a_max, v_max)
            teams.append(sim_rugby.Team(num_players, strat))

        season_score = play_season(teams, dt, time_limit, player_radius, fieldx, fieldy, num_players, v_max, a_max)

        return season_score

    return fitness_func

def play_season(teams, dt, time_limit, player_radius, fieldx, fieldy, num_players, v_max, a_max):
    wins           = np.zeros(len(teams))
    ties           = np.zeros(len(teams))
    losses         = np.zeros(len(teams))
    points_for     = np.zeros(len(teams))
    points_against = np.zeros(len(teams))

    # round robin home and away
    for i in range(len(teams)):
        for j in range(len(teams)):
            if i == j:
                continue

            state0 = sim_rugby.State(num_players, player_radius, v_max, a_max, fieldx, fieldy)

            display = False

            score, state_list = rugby_game.simulate_game(teams[i], teams[j], dt, time_limit, \
                                                         start_state=state0, display=display)

            if display:
                anim = rugby_game.get_animation(state_list, dt)
                anim_name = 'rugby_'+str(i)+'_vs_'+str(j)
                # save animation

            if score[0] > score[1]:
                wins[i]   = wins[i]   + 1
                losses[j] = losses[j] + 1
            elif score[1] > score[0]:
                wins[j]   = wins[j]   + 1
                losses[i] = losses[i] + 1
            else:
                ties[i] = ties[i] + 1
                ties[j] = ties[j] + 1

            points_for[i] = points_for[i] + score[0]
            points_for[j] = points_for[j] + score[1]

    # use season results to rank
    season_score = 3.0*wins + 1.0*ties + points_for / np.sum(points_for)

    return season_score