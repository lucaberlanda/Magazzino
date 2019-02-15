import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
simple_case = False
n = 1000

if simple_case:
    a = pd.Series(range(0, n))
    b = a.apply(lambda x: 1 if x > n/2 else 0)
    c = b * a

    plt.scatter(a, b)
    plt.show()
    plt.scatter(a, c)
    plt.show()
    print(a.corr(b))
    print(a.corr(c))
    quit()

aa = pd.DataFrame(np.random.randn(n, 2))
bb = pd.Series(np.random.choice([0, 1], n))
cc = pd.concat([aa, bb], axis=1)
cc.columns = ['serie1', 'serie2', 'dummy']

ee = pd.concat([cc.loc[:, 'serie1'], pd.concat([cc[cc.serie1 < 0].loc[:, 'serie1'],
                                                cc[cc.serie1 >= 0].loc[:, 'serie2']])], axis=1)

ee.plot.scatter('serie1', 0)
print(ee.expanding.corr())

plt.show()
dd = pd.concat([cc[cc.dummy == 0].loc[:, 'serie2'], cc[cc.dummy == 1].loc[:, 'serie1']])
correlation = pd.concat([dd, cc.loc[:, 'serie1']], axis =1).corr()  # corr of two variables correlated 1/2 of the times
print(correlation)
