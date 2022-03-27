import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Finance.Portfolio.stocks import Stock
from primitive import log_log_plot_with_threshold


def get_n_colors(palette='brg', n=3):
    import matplotlib
    from pylab import cm
    cmap = cm.get_cmap(palette, n)  # PiYG
    colors_list = []
    for i in range(cmap.N):
        rgba = cmap(i)
        colors_list.append(matplotlib.colors.rgb2hex(rgba))  # rgb2hex accepts rgb or rgba

    return colors_list


def plot_drawdown(dist):
    """
    log of the minimum of S over a window of n days following a given S.
    :param dist:
    :return:
    """

    dist = dist.apply(lambda x: None if x >= 0 else x)
    dist = dist.dropna().reset_index()
    dist.columns = ['index', 'rets']
    dist['days_passed'] = (dist['index'] - dist['index'].shift()).shift(-1)

    dd_groups_list = []
    m = 0
    for i in dist.index:
        if dist.loc[i, 'days_passed'] == 1:
            m += 1
        else:
            to_store = dist.iloc[(i - m):(i + 1), :]
            dd_groups_list.append(to_store)
            m = 0

    dd_dict = {}
    for dd_series, counter in zip(dd_groups_list, range(len(dd_groups_list))):
        drawdown_rets = [1] + (dd_series.loc[:, 'rets'] + 1).values.tolist()
        drawdown = pd.Series(drawdown_rets).cumprod().values[-1] - 1
        dd_dict[counter] = drawdown

    dd_distrib_to_plot = pd.Series(dd_dict).sort_values().reset_index().drop('index', axis=1).reset_index()
    dd_distrib_to_plot.columns = ['rank_order', 'drawdown']
    dd_distrib_to_plot.sort_values('rank_order', ascending=False)

    fig = plt.figure(1, figsize=(8, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_xlim(max(dd_distrib_to_plot['rank_order']))
    ax1.invert_yaxis()
    ax1.invert_xaxis()

    dd_distrib_to_plot.plot('rank_order', 'drawdown', kind='scatter', s=20,
                            alpha=1, logx=True, ax=ax1, marker='x', c='black')

    dd_distrib_to_plot.loc[:, 'drawdown'] = dd_distrib_to_plot.loc[:, 'drawdown'] * (-1)
    dd_mean = dd_distrib_to_plot.loc[:, 'drawdown'].mean()
    dd_distrib_to_plot['theoretical_frequency'] = pd.Series \
        (1 - np.power(np.e, ((-1 / dd_mean) * dd_distrib_to_plot.loc[:, 'drawdown'])))

    dd_distrib_to_plot['rank_order'] = (dd_distrib_to_plot.loc[:, 'rank_order'] / max(dd_distrib_to_plot.index))
    dd_distrib_to_plot.columns = ['frequency', 'drawdown', 'theoretical_frequency']
    dd_distrib_to_plot['theoretical_frequency'] = 1 - dd_distrib_to_plot['theoretical_frequency']

    # PLOT

    dd_distrib_to_plot.plot('frequency', 'drawdown', kind='scatter',
                            logx=True, s=30, alpha=1, ax=ax2, marker='x', c='black')

    plt.plot(dd_distrib_to_plot.loc[:, 'theoretical_frequency'],
             dd_distrib_to_plot.loc[:, 'drawdown'], c='red')

    ax2.invert_yaxis()
    ax2.invert_xaxis()
    plt.tight_layout()
    plt.show()


def plot_drawdown_by_window(ris, n_list: list, threshold=0.02):
    """
    log of the minimum of S over a window of n days following a given S.
    :param dist:
    :return:
    """

    dds_by_n = {}
    for n in n_list:
        dds = {}
        for _, df in ris.groupby(np.arange(len(ris)) // n):
            dds[df.index[-1]] = np.log(df.div(df.iloc[0])).min()

        dds_by_n[n] = pd.Series(dds)

    c_list = get_n_colors(n=len(dds_by_n.keys()))
    for cnt, k in enumerate(dds_by_n.keys()):
        s = dds_by_n[k]
        to_plot = abs(s.dropna()).sort_values(ascending=False).reset_index().iloc[:, 1].reset_index()
        to_plot.columns = ['p_>_mod_x', 'daily_return']
        to_plot.loc[:, 'p_>_mod_x'] = (to_plot.loc[:, 'p_>_mod_x'] + 1) / (len(to_plot.index) + 1)
        to_plot = to_plot[to_plot.daily_return > threshold]
        from Viz.charting import generate_ax
        ax = generate_ax('Drawdowns by Window (non overlapping)', '$x$', 'Probability of being > $|x|$')
        ax.plot(to_plot['daily_return'],
                to_plot['p_>_mod_x'],
                'o',
                c=c_list[cnt],
                alpha=0.5,
                markeredgecolor='none')

    ax.legend(n_list)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.show()