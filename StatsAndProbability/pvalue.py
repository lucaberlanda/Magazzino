import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

"""
Replication of "A Short Note on P-Value Hacking"
https://arxiv.org/pdf/1603.07532.pdf
"""

sample_size = 5
significance_level = 0.05
std_dev = 1
mean_1 = 0
mean_2 = 0.55
simulations = 100000

true_tstat = (mean_2 - mean_1) / (std_dev / np.sqrt(sample_size))
true_p_value = 1 - norm.cdf(true_tstat)
print('True p-value:', true_p_value)

p_values_dicts = {}
for i in range(simulations):
    realized = np.random.normal(loc=mean_2, scale=std_dev, size=sample_size)
    mean_2_realized = realized.mean()
    std_2_realized = realized.std()
    real_p_value = (mean_2_realized - mean_1) / (std_dev / np.sqrt(sample_size))
    p_values_dicts[i] = real_p_value

pd.Series(p_values_dicts).plot(kind='hist', bins=100)
plt.show()

obs_p_values = pd.Series(1 - norm.cdf(pd.Series(p_values_dicts)))
obs_p_values.plot(kind='hist', bins=100)
obs_p_values_lower_real = np.float(len(obs_p_values[obs_p_values < (1 - norm.cdf(real_p_value))])) / len(obs_p_values)
portion = len(obs_p_values[obs_p_values < true_p_value]) / simulations
plt.show()

print('Observed p-values mean: ', obs_p_values.mean())
print('Observed p-values median: ', obs_p_values.median())
print('Portion of p-values < than the real one: ', portion)

