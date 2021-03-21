import numpy as np
import pandas as pd
from fbm import fbm

def autocorr(ts, lookback, holdingdays, graph=False):
    """
    :param ts,
    :param lookback
    :param holdingdays
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
