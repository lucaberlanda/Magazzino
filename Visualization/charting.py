import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_ax(title, x_label, y_label):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title, size=18)
    ax.set_ylabel(y_label, size=12)
    ax.set_xlabel(x_label, size=12)
    return ax

# ax = generate_ax('Title', 'x_label', '$y_label^2$')
# pd.Series(np.random.normal(0, 1, 1000)).plot(ax=ax, color=base)
# plt.show()
