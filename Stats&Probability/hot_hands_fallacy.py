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

sample_size = range(4, 20, 1)
streaks = 1
probability = 0.75  # 0.5 like coin tossing

success_rate_list = []
success_rate_list_over_n = []

for j in sample_size:
    for i in range(1000):
        sample = np.random.uniform(size=j)
        sample = pd.Series((sample > probability) * 1)
        # print(sample)
        # sample = pd.Series([1, 1, 1])

        sample_roll = sample.rolling(window=streaks).sum()
        picked_draws = pd.concat([sample, sample.shift(-1), sample_roll], axis=1).dropna()
        picked_draws.columns = ['outcome', 'outcome_lag', 'cumulated_sum']
        #print(picked_draws)
        picked_draws = picked_draws[picked_draws.cumulated_sum >= streaks]
        picked_draws = picked_draws.loc[:, 'outcome_lag'].values.tolist()

        picked_draws = [x for x in picked_draws if not np.isnan(x)]
        success = [x for x in picked_draws if x == 1 and x != np.nan and not np.isnan(x)]

        try:
            success_rate = np.float64(len(success)) / np.float64(len(picked_draws))
            #print('')
            #print(success_rate)
        except ZeroDivisionError:
            success_rate = np.nan

        success_rate_list.append(success_rate)

    success_rate_list_over_n.append(pd.Series(success_rate_list).mean())

pd.Series(success_rate_list_over_n).plot()
plt.show()