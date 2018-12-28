import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as sc
import numpy as np
import seaborn as sns
import sklearn as skl
import random

sns.set_style("whitegrid")


rnd_x = pd.Series(np.random.randn(1000)/100)
sum_x = rnd_x.cumsum()
rnd_x  = rnd_x  + 1
path_x = rnd_x.cumprod()
skl.preprocessing.scale(path_x).plot()
skl.preprocessing.scale(sum_x).plot()
plt.show()
print(path_x)
quit()

class MaximaDistribution:

    def simulate_distr(self):

        ts = None

        if self.distr_type == 't_stud':
            ts = np.random.standard_t(1, self.size)

        elif self.distr_type == 'normal':
            ts = np.random.standard_normal(self.size)

        elif self.distr_type == 'exp':
            ts = np.random.standard_exponential(self.size)

        elif self.distr_type == 'gamma':
            ts = np.random.standard_gamma(self.size)

        return ts

    def __init__(self, distr_type, size=10000):

        self.size = size
        self.distr_type = distr_type
        self.simulation = pd.Series(self.simulate_distr())
        self.maximum = self.simulation.expanding().max()
        self.maximum = pd.concat([self.maximum, self.maximum.shift(1)], axis=1)
        self.maximum['new_obs'] = self.maximum.apply(lambda x: 1 if x[0] > x[1] else 0, axis=1)
        self.maximum['cumulative_new_obs'] = self.maximum['new_obs'].cumsum(axis = 0)
        self.max_observations = self.maximum['cumulative_new_obs']
        """
        print(self.maximum)
        quit()
        self.maximum.plot()
        plt.show()
        quit()
        """

simulations = 100
aaa = {}
for i in range(simulations):
    print(i)
    bbb = MaximaDistribution('t_stud').max_observations
    aaa[i] = bbb

ccc = pd.DataFrame(aaa)
ccc.plot()
plt.show()