import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random

sns.set_style('white')
mean_dict = {}
std_dict = {}

for i in np.arange(10000):
    sample = pd.Series(np.random.standard_normal(50))
    mean_exp = sample.expanding().mean()
    std_exp = sample.expanding().std()
    mean_dict[i] = mean_exp.values.tolist()[-1]
    std_dict[i] = std_exp.values.tolist()[-1]

normalized_mean = (pd.Series(mean_dict) - pd.Series(mean_dict).mean()) / pd.Series(mean_dict).std()
normalized_std = (pd.Series(std_dict) - pd.Series(std_dict).mean()) / pd.Series(std_dict).std()
# normalized_mean = pd.Series(mean_dict)
# normalized_std = pd.Series(std_dict)
normalized_mean.plot(kind='hist', alpha=0.5, bins=100)
normalized_std.plot(kind='hist', alpha=0.5, color='black', bins=100)
plt.show()

skew_exp = sample.expanding().skew()
kurt_exp = sample.expanding().kurt()

moments = pd.concat([mean_exp, var_exp, skew_exp, kurt_exp], axis=1)
moments.columns = ['mean', 'var', 'skew', 'kurt']
moments.plot()
plt.show()
plt.close()


print()