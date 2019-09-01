import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

"""

From Sornette, "Why stock Markets Crash"
Difference between Gaussian and Exponential returns' distribution

"""


def plot_frequency_scatter(sim_rets, ax, c='black'):
    sim_rets = abs(sim_rets).sort_values(ascending=False).reset_index().drop('index', axis=1).reset_index().sort_values(0)
    sim_rets.columns = ['distribution_function', 'returns']
    # take out last return (log(0) and plot)
    sim_rets.iloc[:-1,:].plot(kind='scatter', x='returns', y='distribution_function',
                              logy=True, ax=ax, c=c, s=70, alpha=0.5)


ax = plt.subplot()
np.random.seed(0)

sim_rets1 = pd.Series(np.random.exponential(0.01, 1000))
sim_rets2 = pd.Series(np.random.normal(0, 0.01, 1000))
sim_rets3 = pd.Series(np.random.standard_t(2, 1000))
plot_frequency_scatter(sim_rets1, ax, c='green')
plot_frequency_scatter(sim_rets2, ax, c='red')
plot_frequency_scatter(sim_rets3, ax, c='black')

plt.show()