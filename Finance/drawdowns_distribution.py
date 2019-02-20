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
    dd_distrib_to_plot['index'] = dd_distrib_to_plot.loc[:, 'index'] + 1
    dd_distrib_to_plot.columns = ['frequency', 'drawdown']
    dd_distrib_to_plot.plot('drawdown', 'frequency', kind='scatter', logy=True, s=50, alpha=0.5)
    plt.show()


n = 1000
ndq = pd.Series(np.random.normal(0, 0.01, n))
ndq = quandl.get("NASDAQOMX/COMP-NASDAQ", trim_start='1990-03-01', trim_end='2018-04-03').loc[:, 'Index Value']
ndq_rets = ndq.pct_change().dropna()
ndq_rets2 = pd.Series(ndq_rets.values, index=ndq_rets.index)
# aaa = pd.Series(np.random.pareto(1, size=n)).pct_change()
plot_drawdown(ndq_rets2)

