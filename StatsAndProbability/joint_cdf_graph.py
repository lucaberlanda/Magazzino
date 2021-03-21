import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# This import registers the 3D projection, but is otherwise unused.

t = []
q = []
sample = np.random.uniform(0, 1, 100000)
for i in sample:
    t.append(np.random.geometric(i))
    q.append(i)

outcomes = pd.DataFrame([t, q]).T
outcomes.columns = ['t', 'q']
lines = []
for i in range(1, 50):
    aa = outcomes[outcomes.t == i].mean().q
    lines.append(aa)

pd.Series(lines, index=range(1, 50)).plot()
plt.show()
quit()




def create_3d_graph(x, y):
    import matplotlib.pyplot as plt
    import numpy as np

    # Fixing random state for reproducibility
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=40, range=[[0, 4], [0, 4]])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.025, yedges[:-1] + 0.025, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.1    * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

    plt.show()


def create_3d_graph2(xAmplitudes, yAmplitudes):

    x = np.array(xAmplitudes)  # turn x,y data into numpy arrays
    y = np.array(yAmplitudes)

    fig = plt.figure()  # create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')

    # make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    hist, xedges, yedges = np.histogram2d(x, y, bins=(30, 30))
    xpos, ypos = np.meshgrid(xedges[:-1] + xedges[1:], yedges[:-1] + yedges[1:])

    xpos = xpos.flatten() / 2.
    ypos = ypos.flatten() / 2.
    zpos = np.zeros_like(xpos)

    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    dz = hist.flatten()

    # cmap = cm.get_cmap('jet')  # Get desired colormap - you can change this!
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    # rgba = [cmap((k - min_height) / max_height) for k in dz]

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.title("X vs. Y Amplitudes for ____ Data")
    plt.xlabel("My X data source")
    plt.ylabel("My Y data source")
    plt.savefig("Your_title_goes_here")
    plt.show()

"""
def x_var():

    p = np.random.uniform(0, 1)
    print(p)

    if p > 2/3:
        x = np.random.uniform(2, 3)

    elif p < 2/3:
        x = np.random.uniform(1, 2)

    return x


def y_var(x):
    if x < 2:
        y = np.random.uniform(1, 4)
    else:
        y = np.random.uniform(2, 3)

    return y
"""


def x_var():

    p = np.random.uniform(0, 1)
    print(p)

    if p > 3/4:
        x = np.random.uniform(2, 3)

    elif p < 3/4:
        x = np.random.uniform(1, 2)

    return x



def y_var(x):
    if x < 2:
        y = np.random.uniform(1, 4)
    else:
        y = np.random.uniform(2, 3)

    return y


xs = []
ys = []
for i in range(400000):
    x_drawn = x_var()
    xs.append(x_drawn)
    ys.append(y_var(x_drawn))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
pd.Series(xs).hist(ax=ax2, bins=50)
pd.Series(ys).hist(ax=ax1, bins=50)
create_3d_graph(xs, ys)
plt.show()

quit()
aa = pd.DataFrame([xs, ys]).T
aa.columns = ['x', 'y']
k = aa[(aa.x > 2) & (aa.x < 3)]