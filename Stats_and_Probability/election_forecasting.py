
"""

A) Replicate paper on election forecasting [https://arxiv.org/pdf/1703.06351.pdf,
Election Predictions as Martingales: An Arbitrage Approach (Taleb)]
B) Defensive Forecasting []
C) Betting Strategy Evaluation []

- Simulate Brownian Motion


"""

from scipy.stats import logistic

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import Stats_and_Probability.functions_involved as finv


sns.set_style('dark')
bm = finv.brownian(0, 500, 1, 0.2)
X = pd.Series(bm)
Y = pd.Series(logistic.cdf(bm))  # sigmoid of X to bring the process back to [0, 1]

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

X.plot(ax=ax, c='black', linewidth=1)
Y.plot(ax=ax2, c='teal', linewidth=2)
plt.show()
