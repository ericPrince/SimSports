# sim_sports.py
import numpy as np

import sport

# play a single game starting from an initial state
def play_game(state0, dt=None, time_limit=None, list=True):
    # defaults
    dt         = dt         or state0.p.dt
    time_limit = time_limit or state0.p.T

    state_list = []
    t = 0.0
    state = state0.copy()

    if list:
        state_list.append(state)

    while t < time_limit:
        t = t + dt

        state.update(dt)

        if list:
            state_list.append(state)

    return state.score, state_list

# play a whole season of games according to a specific set
# of parms and a set of teams
def play_season(game_parms, teams):
    n = len(teams)
    stats = Season_Stats(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            state0 = sport.State(teams[i], teams[j], game_parms)
            score, state_list = play_game(state0,   \
                                          list=True )
            stats.add_result((i, j), score)

            # do something with state_list??
            # potentially make an animation
            pass

    return stats.season_rankings()

# store and update game results in a season
class Season_Stats:
    def __init__(self, num_teams):
        self.n = num_teams
        self.wins           = np.zeros(num_teams)
        self.ties           = np.zeros(num_teams)
        self.losses         = np.zeros(num_teams)
        self.points_for     = np.zeros(num_teams)
        self.points_against = np.zeros(num_teams)
        self.played         = np.zeros(num_teams)

    def add_result(self, teams, score):
        # update games played
        self.played[teams[0]] = self.played[teams[0]] + 1
        self.played[teams[1]] = self.played[teams[1]] + 1

        # update w/l/t
        if score[0] > score[1]:
            self.wins[   teams[0] ]   = self.wins[   teams[0] ] + 1
            self.losses[ teams[1] ]   = self.losses[ teams[1] ] + 1
        elif score[0] > score[1]:
            self.wins[   teams[1] ]   = self.wins[   teams[1] ] + 1
            self.losses[ teams[0] ]   = self.losses[ teams[0] ] + 1
        else:
            self.ties[teams[0]] = self.ties[teams[0]] + 1
            self.ties[teams[1]] = self.ties[teams[1]] + 1

        # update points for/against
        self.points_for[teams[0]] = self.points_for[teams[0]] + score[0]
        self.points_for[teams[1]] = self.points_for[teams[1]] + score[1]

        self.points_against[teams[0]] = self.points_against[teams[0]] + score[0]
        self.points_against[teams[1]] = self.points_against[teams[1]] + score[1]

    def season_rankings(self):
        return 3.0*self.wins + 1.0*self.ties \
            + ( self.points_for / max(1, self.points_for.sum()) )

    def __add__(a, b):
        new = Season_Stats(a.n)

        new.wins           = a.wins           + b.wins
        new.ties           = a.ties           + b.ties
        new.losses         = a.losses         + b.losses
        new.points_for     = a.points_for     + b.points_for
        new.points_against = a.points_against + b.points_against

        return new