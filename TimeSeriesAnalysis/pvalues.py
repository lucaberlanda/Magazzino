import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet


sns.set_style('white')

sample_size = 10
significance_level = 0.05
std_dev = 1
mean_1 = 0
mean_2 = 0.72

real_p_value = (mean_2 - mean_1) / (std_dev/np.sqrt(sample_size))  # sqrt(n) already included in the ratio
print(1 - norm.cdf(real_p_value))  # f(E(x)), different from the E(f(x))

p_values_dicts = {}
for i in range(10000):
    realized = np.random.normal(loc=mean_2, scale=std_dev, size=5)
    mean_2_realized = realized.mean()
    std_2_realized = realized.std()
    realized_p_value = (mean_2_realized - mean_1) / (std_dev / np.sqrt(sample_size))
    p_values_dicts[i] = realized_p_value

p_values = pd.Series(p_values_dicts)
obs_p_values = pd.Series(1 - norm.cdf(p_values))
expected_p_value = obs_p_values.mean()  # E(f(x)), i.e. expected p-value
ax = obs_p_values.plot(kind='hist', bins=100)
ax.axvline(expected_p_value, color='k', linestyle='--')
obs_p_values_lower_real = np.float(len(obs_p_values[obs_p_values < expected_p_value]))/len(obs_p_values)

print('true mean: ' + str(expected_p_value))
print('percentage of observation < than true mean: ' + str(obs_p_values_lower_real))
plt.show()
