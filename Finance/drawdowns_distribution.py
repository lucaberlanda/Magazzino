import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_drawdown(dist):
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

    ax = plt.subplot(121)
    ax.set_xlim(max(dd_distrib_to_plot['rank_order']))
    ax.invert_yaxis()
    ax.invert_xaxis()

    dd_distrib_to_plot.plot('rank_order', 'drawdown', kind='scatter', s=20,
                            alpha=1, logx=True, ax=ax, marker='x', c='black')

    dd_distrib_to_plot.loc[:, 'drawdown'] = dd_distrib_to_plot.loc[:, 'drawdown'] * (-1)
    dd_mean = dd_distrib_to_plot.loc[:, 'drawdown'].mean()
    dd_distrib_to_plot['theoretical_frequency'] = pd.Series\
        (1 - np.power(np.e, ((-1/dd_mean) * dd_distrib_to_plot.loc[:, 'drawdown'])))

    dd_distrib_to_plot['rank_order'] = (dd_distrib_to_plot.loc[:, 'rank_order'] / max(dd_distrib_to_plot.index))
    dd_distrib_to_plot.columns = ['frequency', 'drawdown', 'theoretical_frequency']
    dd_distrib_to_plot['theoretical_frequency'] = 1 - dd_distrib_to_plot['theoretical_frequency']

    # PLOT
    ax2 = plt.subplot(122)
    dd_distrib_to_plot.plot('frequency', 'drawdown', kind='scatter',
                            logx=True, s=30, alpha=1, ax=ax2, marker='x', c='black')

    plt.plot(dd_distrib_to_plot.loc[:, 'theoretical_frequency'],
             dd_distrib_to_plot.loc[:, 'drawdown'], c='red')

    # ax.invert_yaxis()
    # ax.invert_xaxis()

    plt.show()


n = 1000
ndq = pd.Series(np.random.normal(0, 0.01, n))
ndq = quandl.get("NASDAQOMX/COMP-NASDAQ", trim_start='1990-03-01', trim_end='2018-04-03').loc[:, 'Index Value']
print(ndq.head())
ndq_rets = ndq.pct_change().dropna()
ndq_rets2 = pd.Series(ndq_rets.values, index=ndq_rets.index)
# aaa = pd.Series(np.random.pareto(1, size=n)).pct_change()
plot_drawdown(ndq_rets2)


