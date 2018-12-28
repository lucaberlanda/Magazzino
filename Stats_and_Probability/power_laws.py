import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl


class TailRisk:
    def __init__(self, data_to_download={'distr': 't_student', 'param': 2}, draws=5000):

        self.data_to_download = data_to_download

        if self.data_to_download['distr'] == 't_student':
            self.distribution = pd.Series(np.random.standard_t(data_to_download['param'], draws))

        elif self.data_to_download['distr'] == 'gaussian':
            self.distribution = pd.Series(np.random.normal(size=draws))
        else:
            self.distribution = data_to_download['param']

    @staticmethod
    def plot_histogram(x, degrees=None, sample=None, x_label='MAD on SD', y_label='Frequency'):

        x_label = x_label
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)
        ax.text(2, 0.4, 't-student, ' + str(degrees) + ' degrees of freedom, sample size =' + str(sample),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}, fontsize=30)

        ax.hist(x, bins=50, normed=True, color='yellowgreen')
        ax.set_xlabel(x_label, size=30)
        ax.set_ylabel(y_label, size=25)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.show()

    @staticmethod
    def plot_scatter(x, x_label='Unknown', y_label='Degrees of Freedom', log_scale=False):
        """
        :param y_label:
        :param x_label:
        :param x:
        :param log_scale:
        :return: plot a series with its index
        """
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

        if log_scale:
            ax.set_yscale('log')

        ax.scatter(x.index, x, facecolor="none", edgecolor="green", linewidth='2', s=40)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel(y_label, size=20)
        ax.set_ylabel(x_label, size=20)
        plt.show()

    @staticmethod
    def compute_mad_on_std(s, reverse=False):

        MAD = abs(s - s.mean()).mean()
        std_dev = s.std()

        if reverse:
            ratio = std_dev / MAD
        else:
            ratio = MAD / std_dev

        return ratio

    @staticmethod
    def expanding_metric(s, metric='MAD_on_SD', reverse=False):

        expanding_series = {}

        if metric == 'MAD_on_SD':
            expanding_series = s.expanding().apply(lambda x: TailRisk.compute_mad_on_std(x, reverse=reverse))

        elif metric == 'Hills':
            expanding_series = s.expanding(min_periods=30).apply(lambda x: TailRisk.compute_hill_estimator(x))

        elif metric == 'Kurtosis':
            expanding_series = s.expanding(min_periods=30).apply(lambda x: pd.Series(x).kurtosis())

        else:
            print('Which estimator?')
            quit()

        return expanding_series

    @staticmethod
    def rolling_metric(s, metric='MAD_on_SD', reverse=False):

        rolling_series = {}

        if metric == 'MAD_on_SD':
            rolling_series = s.rolling(window=60).apply(lambda x: TailRisk.compute_mad_on_std(x, reverse=reverse))

        elif metric == 'Hills':
            rolling_series = s.rolling(window=60).apply(lambda x: TailRisk.compute_hill_estimator(x))

        elif metric == 'Kurtosis':
            rolling_series = s.rolling(window=60).kurtosis()


        else:
            print('Which estimator?')
            quit()

        return rolling_series

    @staticmethod
    def compute_hill_estimator(s, m=10):
        hill_estimator = 1 / (np.log(s[:m] / s[m]).sum() / m)  # not m + 1 cuz in the denominator, m considers the 0
        return hill_estimator

    @staticmethod
    def log_log_plot_with_threshold(s, threshold=0):

        if type(s) == pd.Series:
            s = pd.DataFrame(s)

        id = s.columns.values.tolist()[0]
        s = s.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        filtered_data = s[abs(s) > threshold]

        sorted_returns = abs(filtered_data).dropna().sort(columns=filtered_data.columns[0]).reset_index().drop(
            'index', 1).reset_index()

        sorted_returns['cdf'] = 1 - (sorted_returns.loc[:, 'index'] / len(sorted_returns.index))
        plt.loglog(sorted_returns.loc[:, id], sorted_returns.loc[:, 'cdf'], 'o', alpha=0.5)
        plt.title('log-log plot')
        plt.grid(True)
        plt.show()

    @staticmethod
    def ME_plot(s, starting_threshold=0):

        if type(s) == pd.Series:
            s = pd.DataFrame(s)

        s = s.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        filtered_data = s[abs(s.iloc[:, 0]) > starting_threshold]

        mean_exceedances_dict = {}
        u = filtered_data.values.flatten().tolist()

        for i in u:
            exceedances = s[s.iloc[:, 0] > i] - i
            mean_exceedances = exceedances.mean().values[0]
            mean_exceedances_dict[i] = mean_exceedances

        to_plot = pd.Series(mean_exceedances_dict).to_frame().reset_index()
        to_plot.columns = ['threshold', 'mean_exceedances']
        to_plot.plot(x='threshold', y='mean_exceedances', kind='scatter', alpha=0.5, s=70).set_title('ME plot')
        plt.show()


def compute_stuff(i):
    hill_estimator_dict = {}
    data = TailRisk(i)

    if type(data.distribution) == pd.DataFrame:
        data.cleaned_series = abs(data.distribution).sort(i, ascending=False).dropna().reset_index().drop('ref_date',
                                                                                                          1).loc[:, i]
    else:
        data.cleaned_series = abs(data.distribution).sort_values(ascending=False)
        data.cleaned_series.index = range(len(data.cleaned_series.index))

    hills_estimator_func_of_k_dict = {}

    for j in range(1, round(len(data.cleaned_series.index) * 0.1)):  # range(1, len(data.cleaned_series.index)):
        hill = TailRisk.compute_hill_estimator(data.cleaned_series, m=j)
        hills_estimator_func_of_k_dict[j] = hill
        print(hills_estimator_func_of_k_dict)

    hills_estimator_func_of_k = pd.Series(hills_estimator_func_of_k_dict)
    plt.plot(hills_estimator_func_of_k, 'bo', label=str(i), alpha=0.5)
    plt.plot(hills_estimator_func_of_k, 'darkslateblue')
    plt.title('Hill estimator')
    plt.legend()
    plt.show()

    TailRisk.ME_plot(data.cleaned_series)
    TailRisk.log_log_plot_with_threshold(data.cleaned_series)

hills_dict = {}
for i in range(1000):
    # Pareto distribution
    a = 4  # shape and mode
    bins = 100
    rnd_pareto = pd.Series((np.random.pareto(a, 10000) + 1)).sort_values(ascending=False)
    hills = TailRisk.compute_hill_estimator(rnd_pareto, m=round(len(rnd_pareto.index) * 0.5))
    hills_dict[i] = hills

pd.Series(hills_dict).plot()
plt.show()

rnd_pareto.hist(bins=bins)
plt.show()

frequency = pd.Series(range(0, len(rnd_pareto), 1))
frequency = frequency.div(len(rnd_pareto))
print(frequency)
ax = plt.subplot(111)

"""
with x axis scale of power of 2, pareto was not a straight line anymore.
without the "m" parameter, it is not a straight line anymore.
"""

ax.loglog(rnd_pareto, frequency, marker='o', c=None)
ax.set_xlim(xmin=min(rnd_pareto), xmax=max(rnd_pareto))
plt.show()
quit()

sns.set_style('white')
