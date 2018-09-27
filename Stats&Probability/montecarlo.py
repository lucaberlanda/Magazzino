import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def example1():
    """

    The experiment assumes that:
    Stock A would either gain +40% or lose -30% at each period and
    Stock B would either lose  -20% or gain +15% at each period
    (randomly)

    :return:
    """
    p = 0.5
    stockA_up = 0.4
    stockA_down = -0.3
    stockB_up = 0.15
    stockB_down = -0.20
    draws=10000

    stockA= pd.Series(np.random.choice([1 + stockA_up, 1 + stockA_down], size=(draws), p=[1-p, p]))
    stockB= pd.Series(np.random.choice([1 + stockB_up, 1 + stockB_down], size=(draws), p=[1-p, p]))
    strat = 0.5 * stockA + 0.5* stockB

    stockA.iloc[0] = 1
    stockB.iloc[0] = 1
    strat.iloc[0] = 1

    stockA_eq = stockA.cumprod()
    stockB_eq = stockB.cumprod()
    strat_eq = strat.cumprod()


    stockA_eq.plot(logy=True, legend=False)
    stockB_eq.plot(logy=True, legend=False)
    strat_eq.plot(logy=True, legend=False)
    plt.savefig('demo.png', transparent=True)
    print('ciao')

results_dict = {}
"""
for i in range(1000):
    a = pd.Series(np.random.choice([2, 0.5], size=(5), p=[0.5, 0.5]))
    a.iloc[0] = 1
    results_dict[i] = a.cumprod()


pd.DataFrame(results_dict).plot()
plt.show()
quit()
print('ciao')
"""
results = []
median = []

for i in range(1000):
    a = pd.Series(np.random.choice([2, 0.5], size=(100), p=[0.5, 0.5]))
    a.iloc[0] = 1
    b = a.cumprod().values.tolist()[-1]
    results.append(b)
    median.append((pd.Series(results).median()))

print(pd.Series(results).mean())
print(pd.Series(results).median())
pd.Series(median).plot()
plt.show()
print('ciao')