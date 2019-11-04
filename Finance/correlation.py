import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
simple_case = False

n = 100
est = []
sigma = 2
for i in range(1000):
    est.append((pd.Series(np.random.normal(0, sigma, n))**2).mean())

print(pd.Series(est).var(), (2*sigma**4)/n)

n = 10
est = []
sigma = 2
for i in range(1000):
    aa = pd.Series(np.random.normal(0, sigma, n))
    est.append(((aa- aa.mean())**2).mean())

print(pd.Series(est).var(), (2*sigma**4)/(n+1))

choices_xx = []
for n in range(1, 10):
    choices_n = []
    for _ in range(10000):
        probs = [1 - (1 / (n ** 2)), 1 / (n ** 2)]
        choice = np.random.choice([1 / n, n], 1, replace=False, p=probs)[0]
        choices_n.append(choice)

    choices_xx.append(pd.Series(choices_n).var())

pd.Series(choices_xx).plot()
plt.show()
quit()


def dividing_a_stick():

    trials_dict = {}
    for i in range(200000):
        single_trial = {}
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, x)
        single_trial['x'] = x
        single_trial['y'] = y
        trials_dict[i] = single_trial

    trials = pd.DataFrame(trials_dict).T
    cond_on_y = trials[(trials.y > 0.8) & (trials.y < 0.9)]
    cond_on_y.hist(column='x', bins=40)
    plt.show()


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
    data = [100, 101, 102.5, 103, 104, 103.2, 106, 107, 106.3, 106.2, 108, 108.5, 108.7, 110, 113.5]
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    pd.Series(data).plot(ax=ax1, style='o-', linewidth=1.5, c='black')
    pd.Series(data).plot(ax=ax2, style='o-', linewidth=1.5, c='black')
    ax2.set_ylim(0, 130)

    ax1.set_ylabel('value')
    ax2.set_ylabel('value')

    ax1.set_xlabel('day')
    ax2.set_xlabel('day')

    ax1.set_title('Rescaled Y Axis')
    ax2.set_title('Not Rescaled Y Axis')

    plt.tight_layout()
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


def graph4():

    ts = pd.read_excel('article_graph.xlsx')
    data1 = ts.loc[:, 'series_1'].rolling(window=3).mean()
    data2 = ts.loc[:, 'series_2'].rolling(window=3).mean()

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax3 = ax2.twinx()

    pd.Series(data1).plot(ax=ax1, style='o-', linewidth=1.5, c='black')
    pd.Series(data2).plot(ax=ax1, style='o-', linewidth=1.5, c='red')

    pd.Series(data1).plot(ax=ax2, style='o-', linewidth=1.5, c='black', logy=True)
    pd.Series(data2).plot(ax=ax3, style='o-', linewidth=1.5, c='red', label='cos(x)')

    ax1.set_ylabel('value')
    ax2.set_ylabel('value in logarithmic scale')
    ax3.set_ylabel('value on secondary axis')

    ax1.set_xlabel('day')
    ax2.set_xlabel('day')
    ax1.set_title('Original Values')
    ax2.set_title('Rescaled Values')

    plt.tight_layout()
    plt.show()


dividing_a_stick()