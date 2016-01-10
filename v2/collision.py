# collision.py
from __future__ import division
from math import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def collision(x, v, r1, r2, m1, m2, R=1.0):
    # resultant distance and velocity
    dx = sqrt((x[2] - x[0])**2 + (x[3] - x[1])**2)
    dv = sqrt((v[2] - v[0])**2 + (v[3] - v[1])**2)
    d = r1 + r2

    # no collision if no relative difference in velocities
    if dv == 0.0:
        return x, v

    # angles
    gammaxy = atan2((x[3] - x[1]), (x[2] - x[0]))
    gammav  = atan2((v[1] - v[3]), (v[0] - v[1]))
    dg = norm_angle(gammaxy - gammav)

    dr = dx * sin(dg) / d

    # no collision if angles are out of range
    if pi/2 < abs(dg) < 3*pi/2 or abs(dr) > 1:
        return x, v

    # collision angle
    alpha = asin(dr)

    # time to collision
    dc = d * cos(dg)
    t = dc - copysign(d * sqrt(1 - dr**2) / dv, dc)

    # collision should have happened
    if t > 0:
        return x, v

    # go back in time to collision - ???
    x2 = x + v*t

    # calculate elastic collision
    a = tan(gammav + alpha)
    mr = m2 / m1
    dvx2 = 2*(v[0] - v[2] + a*(v[1] - v[3])) \
         / ((1 + a**2) * (1 + mr))

    v_el = v + np.array([-mr*dvx2, -a*mr*dvx2, dvx2, a*dvx2])

    # calculate inelastic collision
    vxcm = (m1*v[0] + m2*v[2]) / (m1 + m2)
    vycm = (m1*v[1] + m2*v[3]) / (m1 + m2)

    v_cm = np.array([vxcm, vycm, vxcm, vycm])
    v_inel = (v_el - v_cm) * R + v_cm

    # go forward in time to present
    x2 = x2 - v_inel * t

    return x, v_inel

def wall_collision(x, v, r, xw, yw, R):
    v2 = v
    t  = 0

    dx = np.array([-1, 1])
    dy = -dx

    if x[0] < xw[0] + r and v[0] < 0:
        v2 = R * v * dx
        t = -(x[0] - xw[0] - r) / v[0]

    elif x[0] > xw[1] - r and v[0] > 0:
        v2 = R * v * dx
        t = -(xw[1] - r - x[0]) / v[0]

    elif x[1] < yw[0] + r and v[1] < 0:
        v2 = R * v * dy
        t = -(x[1] - yw[0] - r) / v[1]

    elif x[1] > yw[1] - r and v[1] > 0:
        v2 = R * v * dy
        t = -(yw[1] - r - x[1]) / v[1]

    x2 = x + (v - v2) * t

    return x2, v2

def norm_angle(angle):
    if angle > 2*pi:
        return angle - 2*pi
    elif angle < -2*pi:
        return angle + 2*pi
    return angle

#--------------------------------------
import draw

def main():
    dt = 0.01
    #tmax = 100.0
    tmax = 10.0

    c1 = [1.0, 0.0, 1.0]
    c2 = [0.0, 1.0, 1.0]

    R  = 0.95

    m1 = 1.0
    m2 = 2.0

    r1 = 1.0
    r2 = 1.4

    wx = [- 5.0,  5.0]
    wy = [-10.0, 10.0]

    x1 = np.array([0.0, 8.0])
    x2 = np.array([0.5, 4.5])

    v1 = np.zeros(2)
    v2 = np.zeros(2)

    g  = np.array([0.0, -10.0])
    mu = 0.05

    t = 0.0

    x1_list = []
    x2_list = []

    while t < tmax:
        t = t + dt

        # update velocities
        v1 = v1 * (1 - mu) + g*dt
        v2 = v2 * (1 - mu) + g*dt

        # update positions
        x1 = x1 + v1*dt
        x2 = x2 + v2*dt

        # check for ball collision
        xc = np.hstack([x1, x2])
        vc = np.hstack([v1, v2])
        xc, vc = collision(xc, vc, r1, r2, m1, m2, R)
        x1 = xc[0:2]
        x2 = xc[2:4]
        v1 = vc[0:2]
        v2 = vc[2:4]

        # check for wall collisions
        x1, v1 = wall_collision(x1, v1, r1, wx, wy, R)
        x2, v2 = wall_collision(x2, v2, r2, wx, wy, R)

        #'''
        print 't  : ' + str(t)
        print 'x1 : ' + str(x1)
        print 'x2 : ' + str(x2)
        print 'v1 : ' + str(v1)
        print 'v2 : ' + str(v2)
        print '--------------------'
        #'''

        x1_list.append(x1)
        x2_list.append(x2)

    def animate(i):
        x1 = x1_list[i]
        x2 = x2_list[i]
        draw.draw_rectangle(wx[0], wx[1], wy[0], wy[1])
        draw.draw_circle(x1[0], x1[1], r1, c1, 0.7)
        draw.draw_circle(x2[0], x2[1], r2, c2, 0.7)

    fig = plt.figure()
    plt.axis('scaled')
    plt.axis('off')

    anim = animation.FuncAnimation(fig, animate, \
                           frames=int(tmax / dt), interval=dt)

    #anim.save('test_collision.mp4', fps=1/dt)
    #plt.show()

if __name__ == '__main__':
    main()