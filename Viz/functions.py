import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style('white')


def get_n_colors(palette='brg', n=3):
    import matplotlib
    from pylab import cm
    cmap = cm.get_cmap(palette, n)  # PiYG
    colors_list = []
    for i in range(cmap.N):
        rgba = cmap(i)
        colors_list.append(matplotlib.colors.rgb2hex(rgba))  # rgb2hex accepts rgb or rgba

    return colors_list


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


def log_log_plot_with_threshold(s_dict, title='Visual Identification of Paretianity', threshold=0):

    from Viz.charting import generate_ax

    c_list = get_n_colors(n=len(s_dict.keys()))

    ax = generate_ax(title, x_label='$x$', y_label='Probability of being > $|x|$')

    for cnt, k in enumerate(s_dict.keys()):
        s = s_dict[k]
        to_plot = abs(s.dropna()).sort_values(ascending=False).reset_index().iloc[:, 1].reset_index()
        to_plot.columns = ['p_>_mod_x', 'daily_return']
        to_plot.loc[:, 'p_>_mod_x'] = (to_plot.loc[:, 'p_>_mod_x'] + 1) / (len(to_plot.index) + 1)
        to_plot = to_plot[to_plot.daily_return > threshold]

        ax.plot(to_plot['daily_return'],
                to_plot['p_>_mod_x'],
                'o',
                alpha=0.5,
                markeredgecolor='none',
                color=c_list[cnt])

    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show()
