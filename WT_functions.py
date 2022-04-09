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

    roll_yield_raw = df_grouped['close'].pct_change().shift(-1)
    roll_yield_raw.index = df.index
    df.loc[:, 'roll_yield_raw'] = roll_yield_raw
    df.loc[:, 'roll_yield'] = df['roll_yield_raw'].div(df['delta_months'])

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
    w_df_mask = w_df_mask.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)

    return df, chosen_contracts, price_df_comm, w_df


def backtest_strategy(prices, w, rebalancing='monthly'):
    w = w.copy()
    prices = prices.copy()

    w.columns = w.columns.droplevel(0)
    prices.columns = prices.columns.droplevel(0)

    # create single level column to pass the DataFrame
    w.columns = ['_'.join(map(str, x)) for x in w.columns]
    prices.columns = ['_'.join(map(str, x)) for x in prices.columns]

    if rebalancing == 'monthly':
        weights_at_reb_dt = w.resample('BM').last().fillna(0).copy()
        idx_cls = IndexConstruction(prices, weights_at_reb_dt.T, name='strategy', rebalancing_f='weights')
        idx_cls.get_index()
    elif rebalancing == 'daily':
        idx_cls = IndexConstruction(prices, w.T, name='strategy', rebalancing_f='daily')
        idx_cls.get_index()
    else:
        raise KeyError('Rebalancing string not valid!')

    return idx_cls, idx_cls.idx


def plot_contract(what, df, code, m, y, names_dict):
    """
    :param what: str;
    :param df: pd.DataFrame;
    :param code: str;
    :param m: int; month
    :param y: int; year
    :param names_dict; dict to map codes to commodity names
    """

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    contracts = df[(df.contract_code == code) & (df.mat_month.isin(m)) & (df.mat_year.isin(y))]
    lbls = ['contract_code', 'mat_month', 'mat_year']
    contracts.set_index(lbls, append=True).loc[:, what].unstack(lbls).plot(ax=ax, cmap='brg')
    ax.set_title('{} - {} Futures Plot'.format(names_dict[code], what))
    plt.show()


def plot_future_curve_and_roll_yield(commodity_at_dt):
    """

    :param commodity_at_dt: pd.DataFrame; specific commodity df at a specific date
    :return:
    """

    dt = commodity_at_dt.index[0]
    code = commodity_at_dt.contract_code.values[0]
    commodity_last_trade_dt = commodity_at_dt.loc[commodity_at_dt.index[0]].set_index('last_trade_date').sort_index()
    max_roll_yield = commodity_last_trade_dt.loc[[commodity_last_trade_dt.roll_yield.idxmax()], ['roll_yield']]
    fig = plt.figure(1, figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    commodity_last_trade_dt.close.plot(ax=ax1, title='{} Futures Curve at {}'.format(code, dt.strftime('%Y-%m-%d')))
    commodity_last_trade_dt.roll_yield.plot(ax=ax2, title='Roll Yield')
    max_roll_yield.plot(ax=ax2, linewidth=0, marker='o', color='red')

    ax2.axhline(0, linewidth=1, linestyle='dashed', color='black')

    plt.tight_layout()
    plt.show()


class IndexConstruction:

    def __init__(self, ris, w, name='idx', rescale_w=True, rebalancing_f='daily', shift_periods=1):

        """
        Index Construction
        :param ris: pandas DataFrame;
        :param w: pandas DataFrame; instrument weights should be on the index, dates in the column;
        :param rebalancing_f: str; it can be:
            - 'daily': the portfolio is rebalanced daily. The weights can be a Series or  a DataFrame
            - 'weights': the portfolio is rebalanced according to the dates and weights given by self.w, i.e.
                the portfolio is allowed to drift;

        :param shift_periods: int; number of periods that the weights should be shifted of. Default = 1.
        """

        self.w = w
        self.ris = ris
        self.name = name
        self.rescale_w = rescale_w
        self.rebalancing_f = rebalancing_f
        self.shift_periods = shift_periods

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
        if self.rebalancing_f == 'daily':
            if type(self.w) == pd.Series:
                w_df_raw = pd.DataFrame(zip(*[self.w] * len(self.ris.index)),
                                        index=self.ris.columns, columns=self.ris.index).T

                w_df_raw = w_df_raw.fillna(0).reindex(self.ris.index).ffill().shift(self.shift_periods)
            else:
                w_df_raw = self.w.T.fillna(0).reindex(self.ris.index).ffill().shift(self.shift_periods)

        elif self.rebalancing_f == 'weights':
            w_at_reb_dts = self.w.fillna(0)
            w_df_raw = rebase_at_xs(self.ris.truncate(before=w_at_reb_dts.columns[0]), w_at_reb_dts)

        else:
            raise KeyError('self.rebalancing_f is not valid!')

        if self.rescale_w and self.rebalancing_f != 'simple':
            self.w_df = w_df_raw.div(w_df_raw.sum(axis=1), axis=0)
        else:
            self.w_df = w_df_raw

    def plot_weights(self, heatmap=False):
        if heatmap:
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
            plt.show()

        else:
            fig, ax = plt.subplots()
            self.w_df.dropna(how='all').plot(figsize=(8, 6),
                                             stacked=True,
                                             cmap='viridis',
                                             ax=ax, linewidth=0.9, legend=False)

            ax.set_title('Weights Evolution', fontsize=15)

            if len(self.w_df.columns) > 10:
                ax.legend(fontsize=10).set_visible(False)
            else:
                ax.legend(fontsize=10)

        plt.tight_layout()
        plt.show()


def rebase_at_x(df, at=100):
    if type(df) == pd.Series:
        df = df.dropna()
        df = df / df.iloc[0] * at
    else:
        df = df.apply(lambda x: x / x.dropna().values.tolist()[0] * at)
    return df


def rebase_at_xs(ri, at):
    """
    :param ri: pd.DataFrame;
    :param at: pd.DataFrame; df that has rebalancing dates as columns and instrument ids as index
    """
    multiple_dts = at.columns
    df_dict = {}
    ri_reb = ri.copy()
    ri_reb = ri_reb.reindex(at.index.get_level_values(0).tolist(), axis=1).ffill()
    # ri_reb.loc[:, ri_reb.isna().all()] = 100

    for i in np.arange(len(multiple_dts)):

        if i == 0:
            df = ri_reb.loc[:multiple_dts[i + 1], :].iloc[1:, :].dropna(how='all', axis=1)
        elif i + 1 == len(multiple_dts):
            df = ri_reb.loc[multiple_dts[i]:, :].iloc[1:, :].dropna(how='all', axis=1)
        else:
            df = ri_reb.loc[multiple_dts[i]:multiple_dts[i + 1], :].iloc[1:, :].dropna(how='all', axis=1)

        df_dict[i] = df.apply(lambda x: x / x.dropna().values.tolist()[0] * at.loc[x.name, multiple_dts[i]])

    df_to_go = pd.concat(df_dict)
    df_to_go.index = df_to_go.index.droplevel()
    return df_to_go
