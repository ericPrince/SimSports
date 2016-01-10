# sport.py
from __future__ import division
import math
import numpy as np

import collision

class Team:
    def __init__(self, strategy, n):
        self.strategy = strategy
        self.n = n

    def __call__(self, state, team_num):
        return self.actions(state, team_num)

    def actions(self, state, team_num):
        # flip state representation if away team
        s = state.copy()
        if team_num == 1:
            s = s.flip()

        n = state.teams[0].n

        # convert state position/velocity 
        # into nnet-compatible input
        nnet_in = np.zeros(8*n)
        nnet_in[0  :  n] = s.x[0]
        nnet_in[  n:2*n] = s.x[1]
        nnet_in[2*n:3*n] = s.y[0]
        nnet_in[3*n:4*n] = s.y[1]
        nnet_in[4*n:5*n] = s.vx[0]
        nnet_in[5*n:6*n] = s.vx[1]
        nnet_in[6*n:7*n] = s.vy[0]
        nnet_in[7*n:8*n] = s.vy[1]

        return self.strategy(nnet_in)

class Game_Parms:
    def __init__(self):
        self.X  = 1.0   # field length (half)
        self.Y  = 0.5   # field width  (half)
        self.gx = 0.2   # goal width   (half)

        self.r  = 0.1   # player radius
        self.rb = 0.02  # ball radius

        self.m  = 1.0   # player mass
        self.mb = 0.01  # ball mass

        self.mu = 0.01  # friction coefficient

        self.a = 0.5    # max player acceleration

        self.R = 0.95   # inelastic collision restitution

        self.dt = 0.1   # default time step
        self.T  = 100   # default max game time

class State:
    def __init__(self, team0, team1, game_parms):
        self.teams = [team0, team1]
        self.p = game_parms

        # create initial starting positions
        self._reset_field()

        self.score = [0, 0]

    def update(self, dt=None):
        dt = dt or self.p.dt

        # get acceleration actions from teams
        ax0, ay0 = self.teams[0](self, 0)
        ax1, ay1 = self.teams[1](self, 1)
        ax = [np.array(ax0), np.array(ax1)]
        ay = [np.array(ay0), np.array(ay1)]

        # limit max acceleration
        am = np.sqrt(ax[0]**2 + ay[0]**2) / self.p.a
        am = am[am > 1.0]
        ax[0][am > 1.0] = ax[0][am > 1.0] / am
        ay[0][am > 1.0] = ay[0][am > 1.0] / am

        am = np.sqrt(ax[1]**2 + ay[1]**2) / self.p.a
        am = am[am > 1.0]
        ax[1][am > 1.0] = ax[1][am > 1.0] / am
        ay[1][am > 1.0] = ay[1][am > 1.0] / am

        # apply field friction
        self.vx[0] = self.vx[0] * (1 - self.p.mu) * dt
        self.vy[0] = self.vy[0] * (1 - self.p.mu) * dt
        self.vx[1] = self.vx[1] * (1 - self.p.mu) * dt
        self.vy[1] = self.vy[1] * (1 - self.p.mu) * dt

        # update positions
        self.x[0] = self.x[0] + self.vx[0] * dt
        self.y[0] = self.y[0] + self.vy[0] * dt
        self.x[1] = self.x[1] + self.vx[1] * dt
        self.y[1] = self.y[1] + self.vy[1] * dt

        #update velocities
        self.vx[0] = self.vx[0] + ax[0] * dt
        self.vy[0] = self.vy[0] + ay[0] * dt
        self.vx[1] = self.vx[1] + ax[1] * dt
        self.vy[1] = self.vy[1] + ay[1] * dt

        # apply collisions between players
        for tma in range(2):
            for tmb in range(tma, 2):
                for pa in range(self.teams[tma].n):
                    for pb in range(self.teams[tmb].n):
                        self._collision_pp(tma, pa, tmb, pb)

        # apply player collisions with ball and/or walls
        for tma in range(2):
            for pa in range(self.teams[tma].n):
                self._collision_pb(tma, pa)
                self._wall_collision_p(tma, pa)

        # apply collision with ball and walls
        self._wall_collision_b()

        # check for goal
        if -self.p.gx < self.bx < self.p.gx:
            if self.by < -self.p.Y:
                self.score[0] = self.score[0] + 1
                self._reset_field(1)

            elif self.by > self.p.Y:
                self.score[1] = self.score[1] + 1
                self._reset_field(0)

    def copy(self):
        new = State(self.teams[0], self.teams[1], self.p)

        new.x[0]  = self.x[0].copy()
        new.x[1]  = self.x[1].copy()
        new.vx[0] = self.vx[0].copy()
        new.vx[1] = self.vx[1].copy()

        new.bx  = self.bx
        new.by  = self.by
        new.bvx = self.bvx
        new.bvy = self.bvy

        new.score[0] = self.score[0]
        new.score[1] = self.score[1]

        return new

    def flip(self):
        self.x[0]  = -self.x[0]
        self.x[1]  = -self.x[1]
        self.vx[0] = -self.vx[0]
        self.vx[1] = -self.vx[1]

        self.bx  = -self.bx
        self.bvx = -self.bvx

        temp = self.score[0]
        self.score[0] = self.score[0]
        self.score[1] = temp

        temp = self.teams[0]
        self.teams[0] = self.teams[0]
        self.teams[1] = temp

        return self

    #-------------------------
    # private methods

    def _reset_field(self, tm_posess=0):
        n = self.teams[0].n
        self.x  = [0, 0]
        self.y  = [0, 0]
        self.vx = [np.zeros(n), np.zeros(n)]
        self.vy = [np.zeros(n), np.zeros(n)]

        x_rows = np.empty(n)
        y_rows = np.empty(n)
        rows = 3
        for row in range(rows):
            i2 = (n * (row + 1)) // rows
            i1 = (n * (  row  )) // rows

            x_rows[i1:i2] = (row+1)/(rows+2) * self.p.X * np.ones(i2 - i1)
            y_rows[i1:i2] = np.arange(i2 - i1)
            
        # team with starting posession should start closer to ball
        dx = 0.5 * 1/(rows+2) * self.p.X

        self.x[     tm_posess     ] = x_rows.copy() - dx
        self.x[(tm_posess + 1) % 2] = x_rows.copy() + dx
        self.y[     tm_posess     ] = y_rows.copy()
        self.y[(tm_posess + 1) % 2] = y_rows.copy() # should it be (-)

        self.x[1] = -self.x[1]

        # ball starts in center
        self.bx  = 0.0
        self.by  = 0.0
        self.bvx = 0.0
        self.bvy = 0.0

    def _collision_pp(self, tma, pa, tmb, pb):
        xc = np.array([self.x[tma][pa], \
                       self.y[tma][pa], \
                       self.x[tmb][pb], \
                       self.y[tmb][pb]  \
                     ])
        vc = np.array([self.vx[tma][pa], \
                       self.vy[tma][pa], \
                       self.vx[tmb][pb], \
                       self.vy[tmb][pb]  \
                     ])

        xc, vc = collision.collision(xc, vc,       \
                               self.p.r, self.p.r, \
                               self.p.m, self.p.m, \
                               self.p.R)

        self.x[tma][pa] = xc[0]
        self.y[tma][pa] = xc[1]
        self.x[tmb][pb] = xc[2]
        self.y[tmb][pb] = xc[3]

        self.vx[tma][pa] = vc[0]
        self.vy[tma][pa] = vc[1]
        self.vx[tmb][pb] = vc[2]
        self.vy[tmb][pb] = vc[3]

    def _collision_pb(self, tma, pa):
        xc = np.array([self.x[tma][pa], \
                       self.y[tma][pa], \
                       self.bx,         \
                       self.by          \
                     ])
        vc = np.array([self.vx[tma][pa], \
                       self.vy[tma][pa], \
                       self.bvx,         \
                       self.bvy          \
                     ])

        xc, vc = collision.collision(xc, vc,        \
                               self.p.r, self.p.rb, \
                               self.p.m, self.p.mb, \
                               self.p.R)

        self.x[tma][pa] = xc[0]
        self.y[tma][pa] = xc[1]
        self.bx         = xc[2]
        self.by         = xc[3]

        self.vx[tma][pa] = vc[0]
        self.vy[tma][pa] = vc[1]
        self.bvx         = vc[2]
        self.bvy         = vc[3]

    def _wall_collision_p(self, tma, pa):
        xc = np.array([ self.x[tma][pa],  self.y[tma][pa]])
        vc = np.array([self.vx[tma][pa], self.vy[tma][pa]])

        xw = [-self.p.X, self.p.X]
        yw = [-self.p.Y, self.p.Y]

        xc, vc = collision.wall_collision(xc, vc, self.p.r, \
                                          xw, yw,           \
                                          self.p.R          )

        self.x[tma][pa]  = xc[0]
        self.y[tma][pa]  = xc[1]
        self.vx[tma][pa] = vc[0]
        self.vy[tma][pa] = vc[1]

    def _wall_collision_b(self):
        if -self.p.gx < self.bx < self.p.gx:
            # ball could be in the goal
            return

        xc = np.array([ self.bx,  self.by])
        vc = np.array([self.bvx, self.bvy])

        xw = [-self.p.X, self.p.X]
        yw = [-self.p.Y, self.p.Y]

        xc, vc = collision.wall_collision(xc, vc, self.p.rb, \
                                          xw, yw,            \
                                          self.p.R           )

        self.bx  = xc[0]
        self.by  = xc[1]
        self.bvx = vc[0]
        self.bvy = vc[1]