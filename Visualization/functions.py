import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')


def quartiles(length=2000, n_timeseries=100, save_figure=False, threshold_percentage=0.75):

    prices = pd.DataFrame(np.random.randn(length, n_timeseries) / 100 + 1).cumprod()
    prices = prices / prices.loc[0, :]  # rebase prices
    sorted_prices = prices.sort_values(prices.index.values[-1], axis=1)

    threshold = round(n_timeseries * threshold_percentage)

    plt.figure(1, figsize=(11, 6))
    plt.subplot(111)
    plt.plot(sorted_prices.iloc[:, :threshold], c='black', linewidth=0.8)
    plt.plot(sorted_prices.iloc[:, threshold:], c='teal', linewidth=0.8)
    if save_figure:
        plt.savefig('demo2.png', transparent=True)
    else:
        plt.show()


def feature_transformation():
    np.random()