import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')

"""

Surprised by the Gambler’s and Hot Hand Fallacies? A Truth in the Law of Small Numbers
(http://www.thebigquestions.com/hothand2.pdf)
by Joshua Miller

"""

# 1 is success
# 0 is failure

streaks = 1
sample_size = range(2, 10, 1)
success_rate_list = []
success_rate_list_over_n = []

for j in sample_size:
    print(j)
    for i in range(1000):
        sample = pd.DataFrame(np.random.choice([0, 1], size=[j, 2]))
        sample_flt = sample[sample.loc[:, 0] == 1]

        try:
            success_rate = len(sample_flt[sample_flt.loc[:, 1] == 1].index) / len(sample_flt.index)
        except ZeroDivisionError:
            success_rate = np.nan

        success_rate_list.append(success_rate)

    success_rate_list_over_n.append(pd.Series(success_rate_list).dropna().mean())
    print(success_rate_list_over_n)

pd.Series(success_rate_list_over_n).plot()
plt.show()