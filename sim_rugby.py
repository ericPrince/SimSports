from __future__ import division
import math
import numpy as np

#class Output:
#    pass

class Team:
    def __init__(self, num_players, strategy, a_max=1.0, v_max=1.0, x_max=1.0, y_max=1.0):
        self.size = num_players
        self.strategy = strategy
        self.a_max = a_max
        self.v_max = v_max
        self.x_max = x_max
        self.y_max = y_max

    def actions(self, state, team_num):
        # return a list of acceleraions and pass action
        s = state.copy()
        if team_num == 1:
            s = s.flip()

        n = s.num_players

        nnet_input = np.zeros(8*n)
        nnet_input[0  :  n] = s.x[0]
        nnet_input[  n:2*n] = s.x[1]
        nnet_input[2*n:3*n] = s.y[0]
        nnet_input[3*n:4*n] = s.y[1]
        nnet_input[4*n:5*n] = s.vx[0]
        nnet_input[5*n:6*n] = s.vx[1]
        nnet_input[6*n:7*n] = s.vy[0]
        nnet_input[7*n:8*n] = s.vy[1]

        return self.strategy(nnet_input)

class State:
    def __init__(self, num_players, v_max, a_max, player_radius=1.0, fieldx=100.0, fieldy=50.0):
        self.down = 0
        self.team_possession = 0
        self.player_possess  = 0
        self.elapsed_time = 0

        self.v_max = v_max
        self.a_max = a_max

        self.x  = [np.zeros(num_players), np.zeros(num_players)]
        self.y  = [np.zeros(num_players), np.zeros(num_players)]
        self.vx = [np.zeros(num_players), np.zeros(num_players)]
        self.vy = [np.zeros(num_players), np.zeros(num_players)]

        self.xb  = 0
        self.yb  = 0
        self.vxb = 0
        self.vyb = 0

        self.num_players = num_players
        self.player_radius = player_radius
        self.fieldx = fieldx
        self.fieldy = fieldy

        self.score = [0, 0]

    def copy(self):
        newstate = State(self.num_players, self.v_max, self.a_max, \
                         self.player_radius, self.fieldx, self.fieldy)
        newstate.x[0]  = self.x[0].copy()
        newstate.x[1]  = self.x[1].copy()
        newstate.vx[0] = self.vx[0].copy()
        newstate.vx[1] = self.vx[1].copy()

        newstate.xb  = self.xb
        newstate.yb  = self.yb
        newstate.vxb = self.vxb
        newstate.vyb = self.vyb

        newstate.score[0] = self.score[0]
        newstate.score[1] = self.score[1]

        newstate.down = self.down
        newstate.team_possession = self.team_possession
        newstate.player_possess  = self.player_possess
        newstate.elapsed_time = self.elapsed_time

        return newstate

    def flip(self):
        self.x[0]  = -self.x[0]
        self.x[1]  = -self.x[1]
        self.vx[0] = -self.vx[0]
        self.vx[1] = -self.vx[1]

        self.xb  = -self.xb
        self.vxb = -self.vxb

        return self


    # dist squared between 2 players or player and ball
    def dist_squared(self, p1=-1, p2=-1):
        if p1 == -1:
            return (self.x[1][p2] -    self.xb   ) ** 2 + (self.y[1][p2] -    self.yb   ) ** 2
        elif p2 == -1:
            return (   self.xb    - self.x[0][p1]) ** 2 + (   self.yb    - self.y[0][p1]) ** 2
        else:
            return (self.x[1][p2] - self.x[1][p1]) ** 2 + (self.y[1][p2] - self.y[0][p1]) ** 2

    # update field if a tackle has happened
    def tackle(self):
        self.down += 1
        self.player_possess = 0

        if self.down >= 5:
            self.turnover()

        self.vx = [np.zeros(self.num_players), np.zeros(self.num_players)]
        self.vy = [np.zeros(self.num_players), np.zeros(self.num_players)]

        # reset positions

        if self.team_possession == 0:
            x1 = self.xb
            x2 = self.xb + 0.10 * self.fieldx
            # check for goal line-ish reset
        else:
            x2 = self.xb
            x1 = self.xb - 0.10 * self.fieldx

        self.reset_field_positions(x1, x2)

        self.player_possess = math.floor(self.num_players / 2) # make this random for fun??

    # reset 
    def reset_field_positions(self, x1, x2):
        dy = 0.1 * self.fieldy
        basey = max(dy * self.num_players/2, min(self.yb, self.fieldy - dy * self.num_players/2))

        for p in range(self.num_players):
            yp = dy * (p - self.num_players/2) + basey

            self.x[0][p] = x1
            self.y[0][p] = yp

            self.x[1][p] = x2
            self.y[1][p] = yp

    def collision(self, p1, p2):
        # inelastic restitution coefficient
        R = 0.7

        # positions and velocities
        x1  = self.x[0][p1]
        y1  = self.y[0][p1]
        x2  = self.x[1][p2]
        y2  = self.y[1][p2]

        vx1 = self.vx[0][p1]
        vy1 = self.vy[0][p1]
        vx2 = self.vx[1][p2]
        vy2 = self.vy[1][p2]

        d  = math.sqrt(( x2 -  x1)**2 + ( y2 -  y1)**2)
        dv = math.sqrt((vx2 - vx1)**2 + (vy2 - vy1)**2)

        if dv == 0:
            return

        # angles of movement and collision
        gammaxy = math.atan2((y2 - y1), (x2 - x1))
        gammav  = math.atan2((vy1 - vy2), (vx1 - vx2))
        dg = gammaxy - gammav
        if dg > 2*math.pi:
            dg = dg - 2*math.pi
        elif dg < -2*math.pi:
            dg = dg + 2*math.pi

        dr = d * math.sin(gammaxy - gammav) / (2*self.player_radius)

        # check for no collision
        if abs(dr) > 1 or math.pi/2 < abs(dg) < 3*math.pi/2:
            return

        alpha   = math.asin(dr)
        a = math.tan(gammav + alpha)

        # change in velocity
        dvx2 = 2*(vx1 - vx2 + a*(vy1 - vy2)) / ((1 + a**2) / 2)

        # elastic collision
        vx2_el = vx2 + dvx2
        vy2_el = vy2 + a*dvx2
        vx1_el = vx1 - dvx2
        vy1_el = vy1 - a*dvx2

        vxcm = (vx1 + vx2) / 2
        vycm = (vy1 + vy2) / 2

        # inelastic factor
        vx1_inel = (vx1_el - vxcm) * R + vxcm
        vy1_inel = (vy1_el - vycm) * R + vycm
        vx2_inel = (vx2_el - vxcm) * R + vxcm
        vy2_inel = (vy2_el - vycm) * R + vycm

        # time to collision
        dc = d * math.cos(gammaxy - gammav)
        t = dc - math.copysign(2*self.player_radius*math.sqrt(1 - dr**2) / dv, dc)

        # only update if collision should have happened already
        if t > 0:
            return

        # go back in time to collision point
        x1 = x1 + vx1*t
        y1 = y1 + vy1*t
        x2 = x2 + vx2*t
        y2 = y2 + vy2*t

        # go forward in time to current time
        x1 = x1 - vx1_inel * t
        y1 = y1 - vy1_inel * t
        x2 = x2 - vx2_inel * t
        y2 = y2 - vy2_inel * t

        # update simulation variables
        self.x[0][p1] = x1
        self.y[0][p1] = y1
        self.x[1][p2] = x2
        self.y[1][p2] = y2

        self.vx[0][p1] = vx1_inel
        self.vy[0][p1] = vy1_inel
        self.vx[1][p2] = vx2_inel
        self.vy[1][p2] = vy2_inel

    def catch(self, p1=None, p2=None):
        if p1:
            if self.team_possession == 1:
                # maybe down should actually be set to -1
                self.turnover()
            else:
                pass
            self.player_possess = p1
        elif p2:
            if self.team_possession == 0:
                # maybe down should actually be set to -1
                self.turnover()
            else:
                pass
            self.player_possess = p2
        else:
            # never happens
            return

        self.xb  = self.x[self.team_possession][self.player_possess]
        self.yb  = self.y[self.team_possession][self.player_possess]
        self.vxb = self.vx[self.team_possession][self.player_possess]
        self.vyb = self.vy[self.team_possession][self.player_possess]

    def turnover(self):
        self.down = 0
        self.team_possession = (self.team_possession + 1) % 2

    def update(self, ax1, ay1, ax2, ay2, dt, pass_flag, px, py):
        self.elapsed_time += dt

        #----------------------------------------------------------------------

        # set limit on max velocities (and accelerations??)
        for i in range(self.num_players):
            vratio = math.sqrt(self.vx[0][i]**2 + self.vy[0][i]**2) / self.v_max
            if vratio > 1:
                self.vx[1][i] /= vratio
                self.vy[1][i] /= vratio

            vratio = math.sqrt(self.vx[1][i]**2 + self.vy[1][i]**2) / self.v_max
            if vratio > 1:
                self.vx[2][i] /= vratio
                self.vy[2][i] /= vratio

            aratio = math.sqrt(ax1[i]**2 + ay1[i]**2) / self.a_max
            if aratio > 1:
                self.ax1[i] /= aratio
                self.ay1[i] /= aratio

            aratio = math.sqrt(ax2[i]**2 + ay2[i]**2) / self.a_max
            if aratio > 1:
                self.ax2[i] /= aratio
                self.ay2[i] /= aratio

        #----------------------------------------------------------------------

        # reverse direction for second team? - or do this outside of this function
        ##ax2 = -ax2

        #----------------------------------------------------------------------

        # update positions and velocities
        for i in range(self.num_players):
            self.x[0][i] += self.vx[0][i] * dt
            self.y[0][i] += self.vy[0][i] * dt
            self.x[1][i] += self.vx[1][i] * dt
            self.y[1][i] += self.vy[1][i] * dt

            # which should update first?
            self.vx[0][i] += ax1[i] * dt
            self.vy[0][i] += ay1[i] * dt
            self.vx[1][i] += ax2[i] * dt
            self.vy[1][i] += ay2[i] * dt

        #----------------------------------------------------------------------

        # ball position
        self.xb += self.vxb * dt
        self.yb += self.vyb * dt

        # ball velocity and check for passing and tackle
        if self.player_possess != -1:
            if pass_flag:
                # change position of ball to edge of player
                self.xb += px / self.player_radius
                self.yb += py / self.player_radius

                self.vxb += px
                self.vyb += py

                # check for forward pass
                if (self.team_possession == 0 and self.vxb > 0) or (self.team_possession == 1 and self.vxb < 0):
                    self.tackle()
                    # or turnover??
                    # return
            else:
                self.vxb = self.vx[self.team_possession][self.player_possess]
                self.vyb = self.vy[self.team_possession][self.player_possess]

            # check for tackle
            if self.team_possession == 1:
                for i in range(self.num_players):
                    if self.dist_squared(self.player_possess, i) < self.player_radius ** 2:
                        # reset
                        self.tackle()
            else:
                for i in range(self.num_players):
                    if self.dist_squared(i, self.player_possess) <= self.player_radius ** 2:
                        # reset
                        self.tackle()
        else:
            # keep same pass velocity, or have it slow down
            pass

        #----------------------------------------------------------------------

        # update player collisions
        for p1 in range(self.num_players):
            for p2 in range(self.num_players):
                if self.dist_squared(p1, p2) < self.player_radius ** 2:
                    self.collision(p1, p2)

        # also between team collision???

        #----------------------------------------------------------------------

        # check for out of bounds ball
        if not 0 < self.yb < self.fieldy:
            # reset
            self.tackle()

        #----------------------------------------------------------------------

        # check for catch/interception of pass
        if self.player_possess == -1:
            for p1 in range(self.num_players):
                if self.dist_squared(p1=p1) < self.player_radius ** 2:
                    # possible for catch
                    self.catch(p1=p1)

            for p2 in range(self.num_players):
                if self.dist_squared(p2=p2) < self.player_radius ** 2:
                    # possible for catch
                    self.catch(p2=p2)

        #----------------------------------------------------------------------

        # check for try or goal line drop out
        if  self.xb < 0:
            if self.team_possession == 1:
                self.score[1] = self.score[1] + 4
                # should it be a turnover?
                self.turnover()
                self.reset_field_positions(0.1 * self.fieldx, 0.4 * self.fieldx)
            else:
                # points for drop out?
                self.turnover()
                self.reset_field_positions(0.6 * self.fieldx, 0.9 * self.fieldx)

        elif self.xb > self.fieldx:
            if self.team_possession == 0:
                self.score[1] = self.score[0] + 4
                self.turnover()
                self.reset_field_positions(0.6 * self.fieldx, 0.9 * self.fieldx)
            else:
                self.turnover()
                self.reset_field_positions(0.1 * self.fieldx, 0.4 * self.fieldx)