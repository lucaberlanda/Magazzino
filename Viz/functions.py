import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('white')


def quartiles(length=2000, n_timeseries=100, save_figure=False, threshold_percentage=0.75):

    prices = pd.DataFrame(np.random.randn(length, n_timeseries) / 100 + 1).cumprod()
    prices = prices / prices.loc[0, :]  # rebase prices
    sorted_prices = prices.sort_values(prices.index.values[-1], axis=1)

    threshold = round(n_timeseries * threshold_percentage)

    plt.figure(1, figsize=(11, 6))
    plt.subplot(111)
    plt.plot(sorted_prices.iloc[:, :threshold], c='black', linewidth=0.8)
    plt.plot(sorted_prices.iloc[:, threshold:], c='teal', linewidth=0.8)
    if save_figure:
        plt.savefig('demo2.png', transparent=True)
    else:
        plt.show()


def log_log_plot_with_threshold(s, threshold=0):
    to_plot = abs(s.dropna()).sort_values(ascending=False).reset_index().iloc[:, 1].reset_index()
    to_plot.columns = ['p_>_mod_x', 'daily_return']
    to_plot.loc[:, 'p_>_mod_x'] = (to_plot.loc[:, 'p_>_mod_x'] + 1) / (len(to_plot.index) + 1)
    to_plot = to_plot[to_plot.daily_return > threshold]

    from Viz.charting import generate_ax

    ax = generate_ax('Visual Identification of Paretianity', '$x$', 'Probability of being > $|x|$')
    ax.plot(to_plot['daily_return'], to_plot['p_>_mod_x'], 'o', c='blue', alpha=0.5, markeredgecolor='none')
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.show()