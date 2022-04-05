import numpy as np
import pandas as pd
from fbm import fbm


def autocorr(ts, lookback, holdingdays, graph=False):
    """
    :param ts,
    :param lookback
    :param holding days
    :param plt=False
    """

    from scipy.stats import pearsonr

    ret_lag = ts / ts.shift(lookback) - 1  # shift backwards
    ret_fut = ts.shift(-holdingdays) / ts - 1  # shift forward
    aligned_ret = pd.concat([ts, ret_lag, ret_fut], axis=1).dropna()
    aligned_ret.columns = ['time_series', 'past_returns', 'future_returns']
    aligned_ret = aligned_ret.loc[:, ['past_returns', 'future_returns']]
    shift_window = lookback if lookback < holdingdays else holdingdays
    mask = np.arange(0, len(aligned_ret.index), shift_window)
    returns_flt = aligned_ret.iloc[mask]
    correlation_list = pearsonr(returns_flt.iloc[:, 0], returns_flt.iloc[:, 1])
    correlation = correlation_list[0]
    pvalue = correlation_list[1]

    if graph:
        print(lookback, holdingdays, 'corr: ', correlation, pvalue)
        aligned_ret.columns = ['past', 'future']
        aligned_ret.plot(kind='scatter', x='past', y='future')

    return correlation, pvalue


class GBM:
    def __init__(self, S0, mu, sigma, T, I):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.I = I

    def get_paths(self):
        paths = np.zeros((self.T + 1, self.I), np.float64)
        paths[0] = self.S0

        for t in range(1, self.T + 1):
            rand = np.random.standard_normal(self.I)
            paths[t] = paths[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * 1 +
                                             self.sigma * np.sqrt(1) * rand)

        return pd.DataFrame(paths)

    def ensemble_average(self):
        paths = np.zeros((self.T + 1), np.float64)
        paths[0] = self.S0

        for t in range(1, self.T + 1):
            rand = 0
            paths[t] = paths[t - 1] * np.exp(self.mu)

        return pd.DataFrame(paths)

    def time_average(self):
        paths = np.zeros((self.T + 1), np.float64)
        paths[0] = self.S0

        for t in range(1, self.T + 1):
            rand = 0
            paths[t] = paths[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * 1 +
                                             self.sigma * np.sqrt(1) * rand)

        return pd.DataFrame(paths)
