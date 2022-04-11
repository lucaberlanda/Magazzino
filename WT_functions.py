import numpy as np
import pandas as pd
import os.path as pth
import matplotlib.pyplot as plt

from time import time
from functools import wraps


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return wrap


@timing
def get_and_tidy_up_data(filename='contracts_prices.csv', filename_info='contracts_info.csv'):
    filepath = pth.join(pth.join('Other', 'WisdomTree'), filename)
    filepath_info = pth.join(pth.join('Other', 'WisdomTree'), filename_info)

    df = pd.read_csv(filepath).set_index('date')
    df_info = pd.read_csv(filepath_info).set_index('contract_code')
    names_mapping = df_info.contract_short_name.to_dict()
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
    df.loc[:, 'last_trade_date'] = pd.to_datetime(df['last_trade_date'], format='%d/%m/%Y')

    df = df.reset_index().sort_values(['date', 'mat_year', 'mat_month'],
                                      ascending=[True, False, False]).set_index('date')

    df = df.merge(df_info, left_on='contract_code', right_index=True)

    # we define the value of the contract by multiplying the current price by the size of the contract
    df['value_USD'] = df.close.mul(df.contract_size)
    df['oi_USD'] = df.oi.mul(df.value_USD).fillna(0)
    df['volume_USD'] = df.volume.mul(df.value_USD).fillna(0)

    lbls = ['contract_code', 'mat_month', 'mat_year']
    # calculate the 22 days moving average (business days to get one full month moving average) and append as a column
    volume_ma = df.groupby(lbls)['volume_USD'].rolling(window=22, min_periods=1).mean()
    df = df.set_index(lbls, append=True).join(volume_ma, rsuffix='_1M_MA').reset_index().set_index('date')
    return df, names_mapping


def single_commodity_optimal_contract(df, volume_threshold=3e7, oi_threshold=1e8):
    df_grouped = df.reset_index().groupby(['date', 'contract_code'])
    delta_days = df_grouped['last_trade_date'].diff().abs().dt.days.shift(-1)
    delta_days.index = df.index
    df.loc[:, 'delta_days'] = delta_days

    # divide the days between two contracts by 30 as representative of month #days
    df.loc[:, 'delta_months'] = delta_days / 30

    # compute roll yield
    roll_yield_raw = df_grouped['close'].pct_change().shift(-1)
    roll_yield_raw.index = df.index
    df.loc[:, 'roll_yield_raw'] = roll_yield_raw
    df.loc[:, 'roll_yield'] = df['roll_yield_raw'].div(df['delta_months'])

    # filter for Open interest and Volume
    df_flt = df[(df.volume_USD_1M_MA > volume_threshold) & (df.oi_USD > oi_threshold)]
    chosen_idx = list(
        df_flt.reset_index().groupby(['date', 'contract_code'])['roll_yield'].idxmax().dropna().values.astype(int))

    chosen_contracts = df_flt.reset_index().iloc[chosen_idx].set_index('date')

    cols = ['contract_code', 'mat_month', 'mat_year']
    price_df_comm = df.set_index(cols, append=True).loc[:, 'close'].unstack(cols)

    # sort the columns properly
    price_df_comm.sort_index(level='mat_year', axis=1, inplace=True)

    w_df_mask = ~chosen_contracts.set_index(cols, append=True).loc[:, 'close'].unstack(cols).isnull()
    w_df = w_df_mask.astype(float)  # convert bool for chosen contract in a float

    # make sure that the weight matrix and price matrix have same indexes
    w_df = w_df.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)

    return df, chosen_contracts, price_df_comm, w_df


def backtest_strategy(prices, w, single_commodity=True):
    w = w.copy()
    prices = prices.copy()

    if single_commodity:
        w.columns = w.columns.droplevel(0)
        prices.columns = prices.columns.droplevel(0)

        # create single level column to pass the DataFrame
        w.columns = ['_'.join(map(str, x)) for x in w.columns]
        prices.columns = ['_'.join(map(str, x)) for x in prices.columns]

        weights_at_reb_dt = w.resample('BMS').first().fillna(0).reindex(prices.index).ffill().bfill()
        idx_cls = IndexConstruction(prices,
                                    weights_at_reb_dt.T,
                                    name='strategy',
                                    rebalancing='daily',
                                    shift_periods=0)
        idx_cls.get_index()

    else:
        idx_cls = IndexConstruction(prices,
                                    w.T,
                                    name='strategy',
                                    shift_periods=0,
                                    rebalancing='monthly')
        idx_cls.get_index()

    return idx_cls, idx_cls.idx


def plot_contract(what, df, code, m, y, names_dict, return_fig=False):
    """
    :param what: str; quantity to plot
    :param df: pd.DataFrame;
    :param code: str; code of the commodity
    :param m: int; month
    :param y: int; year
    :param names_dict; dict to map codes to commodity names
    :param return_fig: bool; if we want to return the fig or just show it
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    contracts = df[(df.contract_code == code) & (df.mat_month.isin(m)) & (df.mat_year.isin(y))]
    lbls = ['contract_code', 'mat_month', 'mat_year']
    contracts.set_index(lbls, append=True).loc[:, what].unstack(lbls).plot(ax=ax, cmap='brg')
    ax.set_title('{} - {} Futures Plot'.format(names_dict[code], what))

    if what == 'volume_USD':
        ax.axhline(3e7, linewidth=1, linestyle='dashed', color='black')
        ax.set_yscale('log')

    if what == 'oi_USD':
        ax.axhline(1e8, linewidth=1, linestyle='dashed', color='black')
        ax.set_yscale('log')

    if return_fig:
        return fig
    else:
        plt.show()


def plot_future_curve_and_roll_yield(commodity_at_dt, return_fig=False):
    """
    :param commodity_at_dt: pd.DataFrame; specific commodity df at a specific date
    :param return_fig: bool; if we want to return the fig or just show it
    """
    dt = commodity_at_dt.index[0]
    code = commodity_at_dt.contract_code.values[0]
    commodity_last_trade_dt = commodity_at_dt.loc[commodity_at_dt.index[0]].set_index(
        'last_trade_date').sort_index()
    max_roll_yield = commodity_last_trade_dt.loc[[commodity_last_trade_dt.roll_yield.idxmax()], ['roll_yield']]
    fig = plt.figure(1, figsize=(10, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    commodity_last_trade_dt.close.plot(ax=ax1,
                                       title='{} Futures Curve at {}'.format(code, dt.strftime('%Y-%m-%d')))
    commodity_last_trade_dt.roll_yield.plot(ax=ax2, title='{} Roll Yield at {}'.format(code, dt.strftime('%Y-%m-%d')))
    max_roll_yield.plot(ax=ax2, linewidth=0, marker='o', color='red')
    ax2.axhline(0, linewidth=1, linestyle='dashed', color='black')
    plt.tight_layout()

    if return_fig:
        return fig

    else:
        plt.show()


class IndexConstruction:

    def __init__(self, ris, w, name='idx', rescale_w=True, shift_periods=1, rebalancing='daily'):

        """
        Index Construction
        :param ris: pandas DataFrame;
        :param w: pandas DataFrame; instrument weights should be on the index, dates in the column;
        :param shift_periods: int; number of periods that the weights should be shifted of. Default = 1.
        """

        self.w = w
        self.ris = ris
        self.name = name
        self.rescale_w = rescale_w
        self.shift_periods = shift_periods
        self.rebalancing = rebalancing
        self.rets = ris.pct_change()

        self.idx = None
        self.w_df = None
        self.w_rets = None

    def get_index(self):

        self.get_weights()
        self.w_rets = self.w_df.mul(self.ris.pct_change())
        daily_rets = self.w_rets.sum(axis=1)
        self.idx = rebase_at_x((daily_rets + 1).cumprod())
        self.idx.name = self.name

    def get_weights(self):
        if self.rebalancing == 'daily':
            w_df_raw = self.w.T.fillna(0).reindex(self.ris.index).ffill().bfill().shift(self.shift_periods)
        else:
            w_df_raw = self.rebase_at_xs(self.ris.truncate(before=self.w.columns[0]), self.w)

        if self.rescale_w:
            self.w_df = w_df_raw.div(w_df_raw.sum(axis=1), axis=0)
        else:
            self.w_df = w_df_raw

    def plot_weights(self, return_fig=False):
        """
        Plot the weights as heatmap
        :param return_fig: bool
        :return:
        """

        import seaborn as sns
        import matplotlib.dates as mdates

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(1, 1, 1)
        to_plot = self.w_df.dropna(how='all')
        sns.heatmap(to_plot, ax=ax, cmap='viridis')
        years = mdates.YearLocator()  # every year
        years_fmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        fig.autofmt_xdate()
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()

    @staticmethod
    def rebase_at_xs(ri, at):
        """
        :param ri: pd.DataFrame;
        :param at: pd.DataFrame; df that has rebalancing dates as columns and instrument ids as index
        """
        multiple_dts = at.columns
        df_dict = {}
        ri_reb = ri.copy()
        ri_reb = ri_reb.reindex(at.index.get_level_values(0).tolist(), axis=1)
        ri_reb.loc[:, ri_reb.isna().all()] = 100

        for i in np.arange(len(multiple_dts)):
            if i == 0:
                df = ri_reb.loc[:multiple_dts[i + 1], :].dropna(how='all', axis=1)
            elif i + 1 == len(multiple_dts):
                df = ri_reb.loc[multiple_dts[i]:, :].dropna(how='all', axis=1)
            else:
                df = ri_reb.loc[multiple_dts[i]:multiple_dts[i + 1], :].dropna(how='all', axis=1)

            df_dict[i] = df.shift().iloc[1:, :].apply(
                lambda x: x / x.dropna().values.tolist()[0] * at.loc[x.name, multiple_dts[i]])

        df_to_go = pd.concat(df_dict)
        df_to_go.index = df_to_go.index.droplevel()
        return df_to_go


def rebase_at_x(df, at=100):
    if type(df) == pd.Series:
        df = df.dropna()
        df = df / df.iloc[0] * at
    else:
        df = df.apply(lambda x: x / x.dropna().values.tolist()[0] * at)
    return df


class Stats:

    def __init__(self, rets):

        self.rets = rets
        self.ts = (self.rets + 1).cumprod()  # compute levels

    @staticmethod
    def max_drawdown(xs):
        """
        compute the drawdown of a given timeseries
        :param xs: pandas timeseries
        """

        _xs = xs.values
        dd_ts = _xs / np.maximum.accumulate(_xs)

        i = np.argmin(dd_ts)  # end of the period
        if i == 0:
            return 0
        else:
            j = np.argmax(_xs[:i])
            st = xs.index[j].strftime('%Y-%m-%d')
            end = xs.index[i].strftime('%Y-%m-%d')
            mdd = abs(_xs[i] / _xs[j] - 1)
            return pd.Series([mdd, st, end], index=['max_dd', 'start', 'end'])

    @staticmethod
    def compute_return(s, annualized=True):
        """
        :param s: pd.Series. Return Indexes ts
        :param annualized: bool;
        """
        s = s.dropna()
        years = (pd.to_datetime(s.index[-1]) - pd.to_datetime(s.index[0])).days / 365
        ret = s.iloc[-1] / s.iloc[0] - 1
        if annualized:
            # returns are annualized geometrically
            ann_ret = (ret + 1) ** (1 / years) - 1
            return ann_ret
        else:
            return ret

    @staticmethod
    def compute_st_dev(s, annualized=True):

        """
        :param s: pd.Series. Returns ts.
        :param annualized: bool;
        """

        freq = pd.infer_freq(s.index)
        if freq == 'B' or freq is None:  # business day frequency
            ann_coeff = 252
        elif freq == 'D':  # calendar day frequency
            ann_coeff = 365
        elif freq == 'BMS':  # start of month business day
            ann_coeff = 12
        else:
            raise KeyError('Specifiy valid frequency!')

        s = s.dropna()
        if annualized:
            std = s.std() * np.sqrt(ann_coeff)
        else:
            std = s.std()

        return std

    def arithmetic_average_ret(self):
        return self.rets.apply(lambda x: x.dropna().mean())

    def max_dd(self):
        dd_data = self.ts.apply(lambda x: self.max_drawdown(x.dropna())).T
        return dd_data

    def summary_stats(self):

        stats_dict = dict()

        stats_dict['Total Return'] = self.ts.apply(lambda x: self.compute_return(s=x, annualized=False))
        stats_dict['Average Annualized Return'] = self.ts.apply(lambda x: self.compute_return(s=x, annualized=True))
        stats_dict['Arithmetic Average Return'] = self.arithmetic_average_ret()

        stats_dict['Std. Deviation'] = self.rets.apply(lambda x: self.compute_st_dev(s=x, annualized=False))
        stats_dict['Annualized Std. Deviation'] = self.rets.apply(lambda x: self.compute_st_dev(s=x, annualized=True))

        stats_dict['Sharpe Ratio'] = stats_dict['Average Annualized Return'] / stats_dict['Annualized Std. Deviation']

        dd_data = self.max_dd()
        stats_dict['Maximum Drawdown'] = dd_data.max_dd
        stats_dict['Maximum Drawdown Start Date'] = dd_data.start
        stats_dict['Maximum Drawdown End Date'] = dd_data.end

        return pd.concat(stats_dict).unstack(level=0).T
