# animation.py
import matplotlib.pyplot as plt

def draw_circle(x, y, r, c, a=1.0, user_fig=None):
    circle = plt.Circle((x, y), r, color=c, alpha=a)

    fig = user_fig or plt.gcf()
    fig.gca().add_artist(circle)

def draw_rectangle(x1, y1, x2, y2, user_fig=None):
    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, \
             edgecolor=[0, 0, 0], fill=False)

    fig = user_fig or plt.gcf()
    fig.gca().add_artist(rect)