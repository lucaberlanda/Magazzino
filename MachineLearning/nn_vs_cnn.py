import string
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

l = string.ascii_lowercase

np.random.seed(1)
plot = False

alphabet = [i[0] + i[1] for i in list(itertools.product(l, l))]

# params
days = 260
n_fd = 100
pos_ret = 0.001
neg_ret = -pos_ret

mkt_rets = pd.Series(np.random.normal(0.0, 0.01, days))
mkt_ri = (mkt_rets + 1).cumprod()

fig = plt.figure(1, figsize=(11, 6))
ax = fig.add_subplot(111)

pos_fund_extra_ret = pd.DataFrame(np.random.normal(pos_ret, 0.001, (days, n_fd)), columns=alphabet[:n_fd])
neg_fund_extra_ret = pd.DataFrame(np.random.normal(neg_ret, 0.001, (days, n_fd)), columns=alphabet[n_fd:2 * n_fd])

pos_fund_ret = pd.DataFrame(pos_fund_extra_ret.values + mkt_rets.values.reshape(pos_fund_extra_ret.shape[0], 1),
                            columns=alphabet[:n_fd])
neg_fund_ret = pd.DataFrame(neg_fund_extra_ret.values + mkt_rets.values.reshape(pos_fund_extra_ret.shape[0], 1),
                            columns=alphabet[n_fd:2 * n_fd])

pos_fund_ri = (pos_fund_ret + 1).cumprod()
neg_fund_ri = (neg_fund_ret + 1).cumprod()

if plot:
    pos_fund_ri.plot(ax=ax, color='green', linewidth=0.5)
    neg_fund_ri.plot(ax=ax, color='red', linewidth=0.5)

Xy_2 = neg_fund_ret.T
Xy_1 = pos_fund_ret.T

Xy_2['dummy'] = 0
Xy_1['dummy'] = 1

Xy = pd.concat([Xy_1, Xy_2])

Xy_train, Xy_test = train_test_split(Xy)
X_train, y_train = Xy_train.drop('dummy', axis=1), Xy_train.loc[:, 'dummy']
X_test, y_test = Xy_test.drop('dummy', axis=1), Xy_test.loc[:, 'dummy']

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mkt_ri.plot(ax=ax, color='black', logy=True)
plt.show()
