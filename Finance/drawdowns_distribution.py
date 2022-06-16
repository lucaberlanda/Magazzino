import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from primitive import get_sp500


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


def plot_drawdown_by_window(ris, windows: list, threshold=0.02):
    """
    log of the minimum of S over a window of n days following a given S.
    :param ris:
    :param windows:
    :param threshold:
    :return:
    """

    dds_by_n = {}
    for n in windows:
        from Omnia.stats import Stats
        dds = Stats(ris).max_dd(rolling=True, window=n, min_p=n, logs=True)[1].iloc[::n].dropna()
        dds_by_n[n] = dds.iloc[:, 0]

    from Viz.functions import log_log_plot_with_threshold
    log_log_plot_with_threshold(dds_by_n, threshold=threshold, title='Drawdowns by Window (non overlapping)')


if __name__ == '__main__':
    ret_idx = get_sp500()
    plot_drawdown_by_window(ret_idx, windows=[5, 100, 252])
