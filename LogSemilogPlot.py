import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
ax = plt.subplot()
distr2 = pd.DataFrame(np.random.power(0.1, 1000)).sort(columns=0, ascending=False).reset_index().drop('index', axis=1)
ax.plot(distr2)
ax.set_yscale('log')
plt.show()

distr = pd.DataFrame(np.random.standard_cauchy(1000)).sort(columns=0, ascending=False).reset_index().drop('index', axis=1)
distr.plot()

simulated_df = distr.sort(columns=0, ascending=False).reset_index().reset_index()
plt.show()
simulated_df['freq'] = simulated_df.loc[:, 'level_0'] / len(simulated_df.index) * 100
simulated_df.plot(kind='scatter', x=0, y='freq')
ax = plt.subplot()
ax.set_yscale("log")
plt.show()