import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

sample_size = 10
significance_level = 0.05
std_dev = 1
mean_1 = 0
mean_2 = 0.55

real_p_value =(mean_2 - mean_1) / (std_dev/np.sqrt(sample_size))
print(1 - norm.cdf(real_p_value))

p_values_dicts = {}
for i in range(100000):
    realized = np.random.normal(loc=mean_2, scale=std_dev, size=5)
    mean_2_realized = realized.mean()
    std_2_realized = realized.std()
    real_p_value = (mean_2_realized - mean_1) / (std_dev / np.sqrt(sample_size))
    p_values_dicts[i] = real_p_value

# pd.Series(p_values_dicts).plot(kind='hist', bins=100)
# plt.show()
obs_p_values = pd.Series(1 - norm.cdf(pd.Series(p_values_dicts)))
# obs_p_values.plot(kind='hist', bins=100)
obs_p_values_lower_real = np.float(len(obs_p_values[obs_p_values <
                                                    (1 - norm.cdf(real_p_value))]))/len(obs_p_values)
print(obs_p_values.mean())
print(obs_p_values.median())
# plt.show()
