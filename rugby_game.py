from __future__ import division
import matplotlib.pyplot as plt

import sim_rugby

def simulate_game(team1, team2, dt, time_limit, start_state=None, display=False):
    state = start_state or sim_rugby.State(num_players, player_radius, fieldx, fieldy)

    elapsed_time = 0

    state_list = []

    while elapsed_time < time_limit:
        [ax1, ay1, pass_flag1, px1, py1] = team1.actions(state, 0)
        [ax2, ay2, pass_flag2, px2, py2] = team2.actions(state, 1)

        if state.team_possession == 0:
            px = px1
            py = py1
            pass_flag = pass_flag1
        else:
            px = px2
            py = py2
            pass_flag = pass_flag2

        elapsed_time = elapsed_time + dt
        state.update(ax1, ay1, ax2, ay2, dt, pass_flag, px, py)

        #display and display_state(state)
        if display:
            state_list.append(state.copy())

    return state.score, state_list

def get_animation(state_list, dt):
    state = state_list[0]
    def init():
        draw_field(state)

    def animate(i):
        display_state(state_list[i])

    fig = plt.figure()
    plt.axis([-0.1*state.fieldx, 1.1*state.fieldx, \
              -0.1*state.fieldy, 1.1*state.fieldy])
    plt.axis('scaled')
    plt.axis('off')

    anim = animation.FuncAnimation(fig, animate, init_func=init, \
             frames=len(state_list), interval=dt)

    return anim

def display_state(state):
    # draw field
    #draw_field(state)

    # draw players
    r = state.player_radius
    for p in range(state.num_players):
        x1 = state.x[0][p]
        y1 = state.y[0][p]
        draw_circle(x1, y1, r, [0, 0, 1], 0.5)

        x2 = state.x[1][p]
        y2 = state.y[1][p]
        draw_circle(x2, y2, r, [1, 0, 0], 0.5)

    # draw ball
    draw_circle(state.xb, state.yb, r/4, [0, 0, 0], 1.0)

def draw_field(state):
    draw_rectangle(0, 0, state.fieldx, state.fieldy)
    draw_vline(state.fieldx/2, 0, state.fieldy)

def draw_circle(x, y, r, c, a=1.0, user_fig=None):
    circle = plt.Circle((x, y), r, color=c, alpha=a)

    fig = user_fig or plt.gcf()
    fig.gca().add_artist(circle)

def draw_rectangle(x1, y1, x2, y2, user_fig=None):
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, \
             edgecolor=[0, 0, 0], fill=False)

    fig = user_fig or plt.gcf()
    fig.gca().add_artist(circle)

def draw_vline(x, y1, y2, user_fig=None):
    line = plt.Line2D((x, x), (y1, y2))

    fig = user_fig or plt.gcf()
    fig.gca().add_artist(circle)