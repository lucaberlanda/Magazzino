import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
ax = plt.subplot()
exponent = 2
xs = pd.DataFrame(np.arange(1, 1000))
ys = xs**exponent

ax.scatter(xs, ys, alpha=0.5, c='black')
ax.set_yscale('log')
ax.set_xscale("log")
plt.show()
ys.plot()
plt.show()

distr2 = pd.DataFrame(np.random.power(0.1, 1000)).sort_values(0,
                                                              ascending=False).reset_index().drop('index', axis=1)

distr = pd.DataFrame(np.random.standard_cauchy(1000)).sort_values(0,
                                                           ascending=False).reset_index().drop('index', axis=1)
distr.plot()

simulated_df = distr.sort_values(0, ascending=False).reset_index().reset_index()
plt.show()

simulated_df['freq'] = simulated_df.loc[:, 'level_0'] / len(simulated_df.index) * 100
simulated_df.plot(kind='scatter', x=0, y='freq')
ax = plt.subplot()
ax.set_yscale("log")
plt.show()