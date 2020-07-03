import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def rebase_at_x(df, at=100):
    if type(df) == pd.Series:
        df = df.dropna()
        df = df / df.iloc[0] * at
    else:
        df = df.apply(lambda x: x / x.dropna().values.tolist()[0] * at)
    return df


def get_dow_jones():
    file_path = os.path.join(ROOT_DIR, 'djindus.xlsx')
    ri = pd.read_excel(file_path, sheet_name='RI').dropna()
    pi = pd.read_excel(file_path, sheet_name='PI').dropna()

    ri = ri.rename(columns={'Name': 'ref_date'})
    ri = ri.set_index('ref_date')
    ri.columns = ['dow_jones']

    pi = pi.rename(columns={'Name': 'ref_date'})
    pi = pi.set_index('ref_date')
    pi.columns = ['dow_jones']

    df = pd.concat([pi.loc[:ri.index[0], :].iloc[:-1, :], ri])
    return df


def log_log_plot_with_threshold(s, threshold=0):
    to_plot = abs(s.dropna()).sort_values(ascending=False).reset_index().drop('ref_date', axis=1).reset_index()
    to_plot.columns = ['p_>_mod_x', 'daily_return']
    to_plot.loc[:, 'p_>_mod_x'] = (to_plot.loc[:, 'p_>_mod_x'] + 1) / (len(to_plot.index) + 1)
    to_plot = to_plot[to_plot.daily_return > threshold]
    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.plot(to_plot['daily_return'], to_plot['p_>_mod_x'], 'o', c='blue', alpha=0.5, markeredgecolor='none')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Probability of being > |X|')
    ax.set_xlabel('Daily Returns')
    ax.set_title('Visual Identification of Paretianity', fontsize=20)
    plt.tight_layout()
    plt.show()


def reshuffled_ts(ts):
    ret = ts.pct_change()
    reshuffled_ret = ret.dropna().sample(frac=1)  # default without replacement
    reshuffled_time_s = rebase_at_x((reshuffled_ret.reset_index().loc[:, ts.name] + 1).cumprod())
    return reshuffled_time_s


def lagging_kurtosis(ts, max_lag=100):
    kurt_dict = {}
    for days in range(1, max_lag):
        kurt = ts[::days].pct_change().kurt() + 3  # since it is excess kurtosis
        kurt_dict[days] = kurt
    return kurt_dict


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


def excess_return_calc(stock_ri, bmk):
    """

    :param stock_ri: pandas Series; the series of the asset
    :param bmk: pandas Series; the series of the benchmark
    :return: the excess return time-series
    """

    min_date = max(stock_ri.index[0], bmk.index[0])
    max_date = min(stock_ri.index[-1], bmk.index[-1])

    stock_ri = stock_ri.truncate(before=min_date, after=max_date)
    bmk = bmk.truncate(before=min_date, after=max_date)

    stock_ret = stock_ri.pct_change().dropna()
    bmk_ret = bmk.pct_change().dropna()

    excess_return = stock_ret - bmk_ret
    excess_return_ts = rebase_at_x((excess_return + 1).cumprod())

    return excess_return_ts
