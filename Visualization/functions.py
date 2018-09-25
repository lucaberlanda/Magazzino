import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')


def quartiles():

    prices = pd.DataFrame(np.random.randn(2000, 100) / 100 + 1).cumprod()
    prices = prices / prices.loc[0, :]  # rebase prices
    sorted_prices = prices.sort_values(prices.index.values[-1], axis=1)

    plt.figure(1, figsize=(13, 10))
    plt.subplot(111)
    plt.plot(sorted_prices.iloc[:, :66], c='gray')
    plt.plot(sorted_prices.iloc[:, 66:], c='blue')
    # plt.show()
    plt.savefig('demo.png', transparent=True)
