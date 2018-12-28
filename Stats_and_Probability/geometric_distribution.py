import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def geometric_distribution(trials_number, binary_tranformation=False, min_periods=1):

    draws = pd.Series(np.random.geometric(0.2, trials_number))
    draws_expanding_mean = pd.expanding_mean(draws, min_periods).dropna()
    draws_expanding_stdev = draws.expanding(min_periods).std().dropna()
    draws_expanding_sum = draws.expanding(min_periods).sum().dropna()

    sns.distplot(draws)
    plt.show()
    # plt.hist(draws, 50, facecolor='green', alpha=0.75)

    ax = plt.gca()
    ax2 = ax.twinx()

    ax.plot(draws_expanding_mean, color='red')
    ax2.plot(draws_expanding_sum)
    ax.set_title('Expanding Mean and Sum', size=20)
    draws_expanding_stdev.plot()
    plt.show()

    if binary_tranformation:
        draws_to_binary = draws.apply(lambda x: 1 if x > 0 else 0)
        draws_to_binary_expanding_mean = pd.expanding_mean(draws_to_binary, min_periods)
        draws_to_binary_expanding_mean.plot()

    plt.show()


geometric_distribution(1000)