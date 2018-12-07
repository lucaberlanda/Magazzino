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

sample_size = range(3, 10, 1)
streaks = 1
probability = 0.5  # 0.5 like coin tossing

success_rate_list = []
success_rate_list_over_n = []

for j in sample_size:
    for i in range(1000):
        sample = pd.Series(np.random.choice([0, 1], size=j))
        sample_roll = sample.rolling(window=streaks).sum()
        picked_draws = pd.concat([sample.shift(-1), sample_roll], axis=1)
        picked_draws.columns = ['outcome', 'cumulated_sum']
        picked_draws_flt = picked_draws[picked_draws.cumulated_sum >= streaks]
        picked_draws_flt = picked_draws_flt.loc[:, 'outcome'].values.tolist()
        picked_draws_flt = [x for x in picked_draws_flt if not np.isnan(x)]
        success = [x for x in picked_draws_flt if x == 1 and x != np.nan and not np.isnan(x)]

        try:
            success_rate = np.float64(len(success)) / np.float64(len(picked_draws_flt))
        except ZeroDivisionError:
            success_rate = np.nan

        success_rate_list.append(success_rate)

    sequence_length_mean = pd.Series(success_rate_list).dropna().mean()
    print(sequence_length_mean)
    success_rate_list_over_n.append(sequence_length_mean)

pd.Series(success_rate_list_over_n).plot()

plt.show()

cur = conn_psycopg_research.cursor()
cur.execute("AAA")
conn_psycopg_research.commit()
to_go.to_sql(table, conn_research, if_exists='append')
