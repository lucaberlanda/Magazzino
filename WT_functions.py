import numpy as np
import pandas as pd
import os.path as pth
import matplotlib.pyplot as plt


def plot_single_contract(df, code, m, y):
    single_contract = df[(df.contract_code == code) & (df.mat_month == m) & (df.mat_year == y)]
    single_contract.close.plot()
    plt.show()


def plot_future_curve_and_roll_yield(commodity_at_dt, dt):
    fig = plt.figure(1, figsize=(10, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    aa = commodity_at_dt.loc[commodity_at_dt.index[0]].set_index('last_trade_date').sort_index()
    aa.close.plot(ax=ax1, title='{} Futures Curve at {}'.format(code, dt.strftime('%Y-%m-%d')))
    aa.roll_yield.plot(ax=ax2, title='Roll Yield')
    ax2.axhline(0, linewidth=1, linestyle='dashed', color='black')
    plt.tight_layout()
    plt.show()


class IndexConstruction:

    def __init__(self, ris, w, name='idx', rescale_w=True, rebalancing_f='daily', fees=0, shift_periods=1):

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
        self.shift_periods = shift_periods
        self.rebalancing_f = rebalancing_f

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
        if self.rebalancing_f == 'simple':
            if type(self.w) == pd.Series:
                w_df_raw = self.w.to_frame().shift(self.shift_periods)
            else:
                w_df_raw = self.w.shift(self.shift_periods)

            w_df_raw = w_df_raw.reindex(self.ris.index).ffill().dropna()
            if type(self.ris) == pd.Series:
                self.ris = self.ris.to_frame()

        elif self.rebalancing_f == 'daily':
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
    ri_reb = ri_reb.reindex(at.index.get_level_values(0).tolist(), axis=1)
    ri_reb.loc[:, ri_reb.isna().all()] = 100

    for i in np.arange(len(multiple_dts)):
        at_dt = at.loc[:, multiple_dts[i]]
        if len(at.index.names) == 2 and 'daily' in at_dt.index.get_level_values(1).tolist():
            to_rescale = at_dt.xs('daily', level=1)
            w_to_rescale = to_rescale.sum()

        if i == 0:
            df = ri_reb.loc[:multiple_dts[i + 1], :].iloc[1:, :].dropna(how='all', axis=1)
        elif i + 1 == len(multiple_dts):
            df = ri_reb.loc[multiple_dts[i]:, :].iloc[1:, :].dropna(how='all', axis=1)
        else:
            df = ri_reb.loc[multiple_dts[i]:multiple_dts[i + 1], :].iloc[1:, :].dropna(how='all', axis=1)

        if len(at.index.names) < 2:
            df_dict[i] = df.apply(lambda x: x / x.dropna().values.tolist()[0] * at.loc[x.name, multiple_dts[i]])
        else:
            df_before_resc = df.apply(lambda x: x / x.dropna().values.tolist()[0]
                                                * at.reset_index(1, True).loc[x.name, multiple_dts[i]])

            df_before_resc.loc[:, to_rescale.index] = to_rescale.to_frame().T.reindex(ri_reb.index).ffill().loc[
                                                      df_before_resc.index, :].mul(df_before_resc.loc[:,
                                                                                   to_rescale.index].sum(1),
                                                                                   0) / w_to_rescale
            df_dict[i] = df_before_resc

    df_to_go = pd.concat(df_dict)
    df_to_go.index = df_to_go.index.droplevel()
    return df_to_go
