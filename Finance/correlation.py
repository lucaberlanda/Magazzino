import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
simple_case = False

def graph1():
    n_sample = 20
    max_n_variables = 1000
    corr_threshold = 0.5
    corr_dict = {}

    for n in np.arange(2, max_n_variables):
        rvs = pd.DataFrame(np.random.randn(n_sample, n))
        corrs = abs(rvs.corr())
        n_spurious = (corrs > corr_threshold).sum().sum() - n
        print(n, n_spurious)
        corr_dict[n] = n_spurious

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('number of variables', fontsize=12)
    plt.ylabel('number of spurious correlations', fontsize=12)
    pd.Series(corr_dict).rolling(window=10).mean().plot(ax=ax)
    plt.show()


def graph2():
    data = [100, 101, 105, 103, 102.5, 101, 106, 105, 104.3, 102, 101.6, 103, 106, 102]
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    pd.Series(data).plot(ax=ax1)
    pd.Series(data).plot(ax=ax2)
    ax2.set_ylim(0, 120)

    ax1.set_ylabel('value')
    ax2.set_ylabel('value')

    ax1.set_xlabel('day')
    ax2.set_xlabel('day')

    plt.show()


def graph3():
    data1 = [100, 101, 105, 103, 102.5, 101, 106, 105, 104.3, 102, 101.6, 103, 106, 102]
    data2 = [200, 223, 198, 188, 100, 101, 105, 103, 102.5, 101, 106, 105, 104.3, 102, 101.6, 103, 106, 102]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    pd.Series(data2).plot(ax=ax1)
    pd.Series(data1).iloc[4:].plot(ax=ax2)

    ax1.set_ylabel('value')
    ax2.set_ylabel('value')

    ax1.set_xlabel('day')
    ax2.set_xlabel('day')

    plt.show()


graph3()