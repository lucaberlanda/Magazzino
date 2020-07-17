import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from primitive import *
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import performance_metrics
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def squeeze_nan(x):
    original_columns = x.index.tolist()
    squeezed = x.dropna()
    squeezed.index = [original_columns[n] for n in range(squeezed.count())]
    return squeezed.reindex(original_columns, fill_value=np.nan)


class RollingDecompose:

    def __init__(self, iot, method, st='01-01-2010', st_roll='01-01-2018', last_date_only=False):

        """
        :param method: str; 'prophet', 'holt-winters'
        :param iot:
        :param st
        :param st_roll

        returns
        ----------

        trend_strength_dict: dictionary;

        trend_df: pd.DataFrame;

        trend: pd.Series;

        """

        pd.plotting.register_matplotlib_converters()

        self.method = method
        self.iot = iot.truncate(before=st)
        self.trend_strength_dict = {}
        self.surprise_dict = {}
        self.MAPE_dict = {}

        if last_date_only:
            to_iter = [self.iot.loc[pd.to_datetime(st_roll):].index[0]]
        else:
            to_iter = self.iot.loc[pd.to_datetime(st_roll):].index

        for cnt, dt in enumerate(to_iter):
            print('Dates count: ', cnt, len(to_iter))
            isin = iot.name
            iot_flt = self.iot.loc[:dt]

            if self.method == 'holt-winters':
                model = ExponentialSmoothing(iot_flt, trend='add', seasonal='add', seasonal_periods=52)
                model_fit = model.fit()
                t_strength = model_fit.slope.values[-1]

            elif self.method == 'hp_filter':
                cycle, trend = sm.tsa.filters.hpfilter(iot_flt, 1600)
                t_strength = self.get_trend_strength(iot_flt, trend)

            elif self.method == 'seasonal_decompose':
                ts_decomp = sm.tsa.seasonal_decompose(iot_flt, model='multiplicative')
                trend = ts_decomp.trend

            elif self.method == 'ARIMA':
                from statsmodels.tsa.arima_model import ARIMA
                arima = ARIMA(iot_flt, order=(10, 0, 0))
                arima = arima.fit()

            elif self.method == 'prophet':

                # Apply Facebook Prophet Forecasting Method
                self.to_fit = iot_flt.reset_index()
                self.to_fit.columns = ['ds', 'y']
                self.m = Prophet(seasonality_mode='multiplicative')
                self.m.fit(self.to_fit)
                future = self.m.make_future_dataframe(periods=10, freq='W')
                self.forecast = self.m.predict(future)
                self.residuals, self.MAPE = self.get_residuals_and_MAPE()
                trend_strength = self.get_trend_strength()

                ci = self.forecast.set_index('ds').loc[:, ['yhat_lower', 'yhat_upper']]
                iot_and_ci = pd.concat([self.iot, ci], axis=1).dropna()
                iot_and_ci['lower_delta'] = iot_and_ci.loc[:, isin] - iot_and_ci.loc[:, 'yhat_lower']
                iot_and_ci['upper_delta'] = iot_and_ci.loc[:, isin] - iot_and_ci.loc[:, 'yhat_upper']
                iot_and_ci['lower_surprise'] = iot_and_ci.loc[:, isin] < iot_and_ci.loc[:, 'yhat_lower']
                iot_and_ci['upper_surprise'] = iot_and_ci.loc[:, isin] > iot_and_ci.loc[:, 'yhat_upper']
                iot_and_ci['surprise'] = iot_and_ci.loc[:, 'lower_delta'] * iot_and_ci.loc[:, 'lower_surprise'] + \
                                         iot_and_ci.loc[:, 'upper_delta'] * iot_and_ci.loc[:, 'upper_surprise']

                if cnt == 0:
                    self.surprise_dict[dt] = iot_and_ci
                else:
                    self.surprise_dict[dt] = iot_and_ci.iloc[-1, :]

                self.trend_strength_dict[dt] = trend_strength
                self.MAPE_dict[dt] = self.MAPE

        self.trend_strength_df = pd.DataFrame(self.trend_strength_dict)
        for i in self.trend_strength_df.columns:
            self.trend_strength_df.loc[i:, i] = np.nan

        self.trend_strength = self.trend_strength_df.apply(squeeze_nan, axis=1).iloc[:, 0]

    def get_residuals_and_MAPE(self):
        df = pd.merge(self.to_fit, self.forecast, on='ds')
        residuals = df['yhat'] - df['y']
        MAPE = abs(residuals / df['y']).mean()
        return residuals, MAPE

    def plot_int_er_trend(self, exc_ret_ts, stength_coeff=2):

        fig = plt.figure(figsize=(10, 8), facecolor='white')
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        self.iot.plot(ax=ax1, linestyle="", marker="o", color='#e85b6e', markersize=3)
        self.iot.plot(ax=ax1, alpha=0.5, color='#e85b6e', linewidth=1)

        # extend to daily prices
        trend_er = pd.concat([exc_ret_ts, self.trend_strength], axis=1).ffill()
        trend_er.columns = ['excess_return', 'trend']
        strength_exp = trend_er.trend
        conviction = strength_exp * stength_coeff
        conviction = conviction.truncate(before=exc_ret_ts.index[0], after=exc_ret_ts.index[-1])
        exc_ret_ts_strat = rebase_at_x(((exc_ret_ts.pct_change() * (1 + conviction)) + 1).fillna(1).cumprod())
        exc_ret_ts.plot(ax=ax2, color='#042c58')
        exc_ret_ts_strat.plot(ax=ax2, color='red')

        pos_signal = self.trend_strength.copy()
        neg_signal = self.trend_strength.copy()

        pos_signal[pos_signal <= 0] = np.nan
        neg_signal[neg_signal > 0] = np.nan
        pos_signal.plot(color='#148957')
        neg_signal.plot(color='#ff2350')

        # strength.plot(ax=ax3, color='#2d4b97')

        ax1.set_title("Stock Interest Over Time")
        ax2.set_title("Excess Return Plot")
        ax3.set_title("Trend Strength")

        plt.tight_layout()
        plt.show()

    def get_trend_strength(self, plot=False):

        if self.method == 'prophet':
            trend_strength = self.forecast.set_index('ds').trend.shift(-1) - self.forecast.set_index('ds').trend
            trend_strength.index = [pd.to_datetime(dt) for dt in trend_strength.index]
        else:
            trend_strength = self.forecast.shift(-1) - self.forecast

        if plot:
            self.forecast.plot()
            self.iot.plot()
            self.trend_strength.plot(secondary_y=True)
            plt.show()

        return trend_strength