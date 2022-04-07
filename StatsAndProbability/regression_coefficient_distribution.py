import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

n_features = 10
n_samples = 60

distr = {'distribution': np.random.normal,
         'params': [[0, 1], [n_samples, n_features]]}
rsq_list = []
for i in range(10000):
    df = pd.DataFrame(distr['distribution'](*distr['params']))
    s = pd.Series(np.random.normal(0, 1, n_samples))

    df = pd.DataFrame(np.random.standard_t(1, [n_samples, n_features]))
    s = pd.Series(np.random.standard_t(1, n_samples))

    result = OLS(endog=s, exog=df).fit()
    rsq_list.append(result.rsquared)

import matplotlib.pyplot as plt

pd.Series(rsq_list).plot(kind='hist', bins=100)
plt.show()
