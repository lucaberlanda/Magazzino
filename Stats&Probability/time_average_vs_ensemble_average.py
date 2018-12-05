import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
References:
- Optimal Leverage From Non Ergodicity, Ole Peters [https://arxiv.org/pdf/0902.2965.pdf]

- Average as a function of time
- How is diffusion process? can go < 0
- Insert theoretical prob 
"""

sns.set_style('white')
save_figure = True
mean_value = 0.1
iterations = 1000


def time_vs_ensemble_average(mean, st_dev, size):
    # np.random.seed(1)
    rets = pd.Series(np.random.normal(mean, st_dev, size))
    ensemble_avg = rets.mean()
    cumulated_rets = (rets + 1).cumprod()
    time_avg = (cumulated_rets.iloc[-1] / cumulated_rets.iloc[0]) ** (1 / len(rets)) - 1
    print('Ensemble average: ' + str(round(ensemble_avg, 4)) + ', theoretical ' + str(round(mean, 4)))
    theoretical_ensemble_avg = mean - ((st_dev**2)/2)
    print('Time average: ' + str(round(time_avg, 4)) + ', theoretical ' + str(round(theoretical_ensemble_avg, 4)))
    return time_avg, ensemble_avg, mean, theoretical_ensemble_avg


collection = {}
for i in np.arange(0.01, 0.4, 0.0001):
    time_and_ensemble_dict = {}
    t_avg, e_avg, te_avg, tt_avg = time_vs_ensemble_average(0.06, i, 1000)

    time_and_ensemble_dict['time'] = t_avg
    time_and_ensemble_dict['ensemble'] = e_avg
    time_and_ensemble_dict['theoretical_time'] = tt_avg
    time_and_ensemble_dict['theoretical_ensemble'] = te_avg

    collection[i] = time_and_ensemble_dict

fig = plt.figure()
ax = fig.add_subplot(111)
to_plot = pd.DataFrame(collection).T

to_plot.loc[:, 'time'].plot(ax=ax, linewidth=0.5, c='red', alpha=0.5, legend=True)
to_plot.loc[:, 'ensemble'].plot(ax=ax, linewidth=0.5, c='black', alpha=0.5, legend=True)
to_plot.loc[:, 'time'].ffill().rolling(window=100).mean().plot(ax=ax, linewidth=3, c='red', legend=False)
to_plot.loc[:, 'ensemble'].rolling(window=100).mean().plot(ax=ax, linewidth=3, c='black', legend=False)
to_plot.loc[:, 'time'].plot(ax=ax, linewidth=0.5, c='red', alpha=0.8)
to_plot.loc[:, 'theoretical_time'].plot(ax=ax, linewidth=1.5, c='red', alpha=1)
to_plot.loc[:, 'ensemble'].plot(ax=ax, linewidth=0.5, c='black', alpha=0.8)
to_plot.loc[:, 'theoretical_ensemble'].plot(ax=ax, linewidth=1.5, c='black', alpha=1)

ax.set_xlabel('Volatility')
ax.set_ylabel('Average')
ax.set_title(r'Time vs Ensemble Average: $\mu=%s$, iterations=%s' % (str(mean_value), iterations))

if save_figure:
    plt.savefig('time_average_vs_ensemble_average.png', transparent=True)

plt.show()
