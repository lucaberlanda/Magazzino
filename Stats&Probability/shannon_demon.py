import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

buy_and_hold_ret_dict = {}
shannon_ret_dict = {}
steps = 100
amount_bet = list(np.arange(0.01, 1, 0.01))
aaa = {}

for i in range(steps):

    print(i)
    returns = np.random.choice([0.5, 2], steps)
    returns[0] = 1
    returns = pd.Series(returns)

    growth_dict = {}
    for j in amount_bet:
        returns2 = returns.copy()

        for k in range(0, steps - 1, 1):
            returns2.loc[k + 1] = (returns2.loc[k] * j) + (returns2.loc[k] * (1 - j)) * returns2.loc[k + 1]

        growth = returns2.values[-1] ** (1 / steps) - 1
        growth_dict[j] = growth

    aaa[i] = pd.Series(growth_dict)

ccc = pd.DataFrame(aaa).stack(level=0).to_frame().reset_index()
ccc = ccc.drop('level_1', axis=1)
ccc.columns = ['cash_level', 'growth_rate']
ccc.plot(y='growth_rate', x='cash_level',  kind='scatter')
plt.savefig('demo.png', transparent=True)
