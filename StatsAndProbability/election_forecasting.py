
"""

A) Replicate paper on election forecasting [https://arxiv.org/pdf/1703.06351.pdf,
Election Predictions as Martingales: An Arbitrage Approach (Taleb)]
B) Defensive Forecasting []
C) Betting Strategy Evaluation []

- Simulate Brownian Motion

"""

from scipy.stats import logistic
from scipy.special import erf, erfinv, erfc, erfcinv

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import StatsAndProbability.functions_involved as finv

sns.set_style('dark')

# params

sigma = 0.05
T = 100
dt = 1
x0 = 0
threshold = 0.5

bm = finv.brownian(x0, T, dt, sigma, seed=True)
X = pd.Series(bm)
Y = pd.Series(logistic.cdf(bm))  # sigmoid of X to bring the process back to [0, 1]
Y2 = pd.Series(0.5 + 0.5 * erf(bm))
B = {}

for i in Y.index:
    v = Y.loc[i]

    num = (threshold - (erfinv(2 * v - 1) * np.exp((sigma ** 2) * (T - i))))
    den = np.sqrt(np.exp(2 * (sigma**2) * (T - i)) - 1)

    print('num: ' + str(num))
    print('den: ' + str(den))
    print('')
    B[i] = 0.5 * erfc(num/den)

B = pd.Series(B)

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

X.plot(ax=ax, c='black', linewidth=1)
Y.plot(ax=ax2, c='teal', linewidth=2)
B.plot(ax=ax2, c='red', linewidth=2)
plt.show()

