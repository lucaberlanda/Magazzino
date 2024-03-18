import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t

from scipy.stats import ttest_1samp
"""
Replication of "A Short Note on P-Value Hacking"
https://arxiv.org/pdf/1603.07532.pdf
"""

sample_size = 15
significance_level = 0.05
std_dev = 1
mean_1 = 0
mean_2 = 0.35
simulations = 10000

true_tstat = (mean_2 - mean_1) / (std_dev / np.sqrt(sample_size))
true_p_value = 1 - t.cdf(true_tstat, df=sample_size-1)
print('True p-value:', true_p_value)
print('True t-statistic:', true_tstat)

p_values_dict = {}
t_stat_dict = {}
for i in range(simulations):
    realized = np.random.normal(loc=mean_2, scale=std_dev, size=sample_size)
    mean_2_realized = realized.mean()
    std_2_realized = realized.std()
    ttest = ttest_1samp(realized, popmean=mean_1)

    real_t_statistic = ttest.statistic
    pvalue = ttest.pvalue

    t_stat_dict[i] = real_t_statistic
    p_values_dict[i] = pvalue

pd.Series(t_stat_dict).plot(kind='hist', bins=100, color='black')
plt.show()

obs_p_values = pd.Series(p_values_dict)
obs_p_values.plot(kind='hist', bins=100)
obs_p_values_lower_real = np.float(len(obs_p_values[obs_p_values < true_p_value])) / len(obs_p_values)
portion = len(obs_p_values[obs_p_values < true_p_value]) / simulations
plt.show()

print('Observed p-values mean: ', obs_p_values.mean())
print('Observed p-values median: ', obs_p_values.median())
print('Portion of p-values < than the real one: ', portion)

