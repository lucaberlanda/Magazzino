import random
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def cauchy_simulation(trials_number, binary_transformation=False, min_periods=1):

    cauchy = pd.Series(np.random.standard_cauchy(trials_number))
    cauchy_expanding_mean = pd.expanding_mean(cauchy, min_periods).dropna()
    cauchy_expanding_stdev = cauchy.expanding(min_periods).std().dropna()
    cauchy_expanding_sum = cauchy.expanding(min_periods).sum().dropna()

    plt.hist(cauchy, 50, facecolor='green', alpha=0.75)
    plt.show()

    ax = plt.gca()
    ax2 = ax.twinx()

    ax.plot(cauchy_expanding_mean, color='red')
    ax2.plot(cauchy_expanding_sum)
    plt.show()
    quit()
    cauchy_expanding_stdev.plot()
    plt.show()

    if binary_transformation:
        cauchy_to_binary = cauchy.apply(lambda x: 1 if x > 0 else 0)
        cauchy_to_binary_expanding_mean = pd.expanding_mean(cauchy_to_binary, min_periods)
        cauchy_to_binary_expanding_mean.plot()

    plt.show()


def three_dimensions_plot_example():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    a = pd.Series(np.random.randn(1000))
    b = pd.Series(np.random.randn(1000))
    c = pd.Series(np.random.randn(1000))

    print(a.tail())
    print(b.tail())
    print(c.tail())

    ax.scatter(a, b, c)
    plt.show()


def simulate_gaussian(mean=0, vol=0.1, scaling_factor=0, n=1):
    sim = np.random.normal(mean, vol * (scaling_factor+1), size=n)
    if len(sim)==1:
        sim = sim[0]

    return sim


def gaussian_scaling_factor(alpha, prob):
    p_aux = prob / (1 - prob)
    beta = -alpha * p_aux
    return beta


def bool_based_on_prob(probability):
    return random.random() < probability


def mixture_of_gaussian(p=1/2000, alpha=1950, mean=0, vol=0.1, number_of_draws=1000, plot=True):

    beta = gaussian_scaling_factor(alpha, p)

    dict = {}  # initialize dictionary to store
    for i in range(number_of_draws):
        dict[i] = bool_based_on_prob(p)

    switching_operator = pd.Series(dict)  # True or False based on the probability (p=10%, True with 10% probability)
    returns_list = []  # initialize returns list

    for switch in switching_operator:
        if switch == True:
            observation = simulate_gaussian(scaling_factor=alpha)
            returns_list.append(observation)
        else:
            observation = simulate_gaussian(scaling_factor=beta)
            returns_list.append(observation)

    returns_series1 = pd.Series(returns_list).to_frame()
    returns_series2 = pd.Series(simulate_gaussian(mean=mean, vol=vol, n=number_of_draws)).to_frame()
    returns_series = pd.concat([returns_series1, returns_series2], axis=1)
    returns_series_std_dev = returns_series.std()

    returns_series.columns = ['series1', 'series2']

    returns_series['equity_line1'] = returns_series['series1'].cumsum()
    returns_series['equity_line2'] = returns_series['series2'].cumsum()

    if plot:
        returns_series.loc[:, ['equity_line1', 'equity_line2']].plot()
        plt.show()

    return returns_series


def gaussian_kernel(mu_list=[0], sigma_list=[1]):

    fig = pl.figure(figsize=(12, 9))
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)

    x = np.linspace(min(mu_list) - max(sigma_list) * 4,
                    max(mu_list) + max(sigma_list) * 4,
                    5000)

    ax1.set_title('Gaussian Distributions', size=30)
    legend_strings = []

    for mu, sigma in zip(mu_list, sigma_list):
        distr = pd.Series((1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) *
                          (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(sigma, 2))))))

        distr.index = x
        legend_str = 'mu = %s, sigma = %s' % (str(mu), str(sigma))
        legend_strings.append(legend_str)
        distr.plot(ax=ax1, legend=legend_str)

    ax1.legend(legend_strings, prop={'size': 15})
    return ax1


def exponential_kernel(scale=[0]):

    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)
    x = np.linspace(0, 5, 5000)

    ax1.set_title('Exponential Distributions', size=15)
    legend_strings = []

    for mu in scale:
        distr = pd.Series(mu * np.power(np.e, -(mu * x)))
        distr.index = x
        legend_str = 'mu = %s' % str(mu)
        legend_strings.append(legend_str)
        distr.plot(ax=ax1, legend=legend_str)

    ax1.legend(legend_strings, prop={'size': 15})
    return ax1


gaussian_plot = exponential_kernel(scale=[1/0.01, 1/0.02, 1/0.03])
plt.savefig('demo.png', transparent=True)



