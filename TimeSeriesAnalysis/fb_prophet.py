import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as smapi
import statsmodels as sm
from primitive import *
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.gam.api import GLMGam, BSplines


def plot_int_er_trend(inter, exc_ret_ts, strength, stength_coeff=2):

    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    inter.plot(ax=ax1, linestyle="", marker="o", color='#e85b6e', markersize=3)
    inter.plot(ax=ax1, alpha=0.5, color='#e85b6e', linewidth=1)

    strength_exp = pd.concat([exc_ret_ts, strength], axis=1).ffill().trend
    conviction = strength_exp * stength_coeff
    conviction = conviction.truncate(before=exc_ret_ts.index[0], after=exc_ret_ts.index[-1])
    exc_ret_ts_strat = rebase_at_x(((exc_ret_ts.pct_change() * (1 + conviction)) + 1).fillna(1).cumprod())
    exc_ret_ts.plot(ax=ax2, color='#042c58')
    exc_ret_ts_strat.plot(ax=ax2, color='red')

    pos_signal = strength.copy()
    neg_signal = strength.copy()

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


def squeeze_nan(x):
    original_columns = x.index.tolist()
    squeezed = x.dropna()
    squeezed.index = [original_columns[n] for n in range(squeezed.count())]
    return squeezed.reindex(original_columns, fill_value=np.nan)


def get_trend_strength(ss, fcst, fb=False, plot=False):

    if fb:
        trend_strength = fcst.set_index('ds').trend.shift(-1) - fcst.set_index('ds').trend
        trend_strength.index = [pd.to_datetime(dt) for dt in trend_strength.index]
    else:
        trend_strength = fcst.shift(-1) - fcst

    if plot:
        fcst.plot()
        ss.plot()
        trend_strength.plot(secondary_y=True)
        plt.show()

    return trend_strength


st = '01-01-2010'
st_roll = '01-01-2018'
iots = pd.read_excel('g_trend.xlsx', sheet_name='interest_over_time')
stocks_ris = pd.read_excel('g_trend.xlsx', sheet_name='stocks_ris')
sp_ri = pd.read_excel('g_trend.xlsx', sheet_name='S&P_ri').set_index('ref_date').truncate(before=st).iloc[:, 0]
etf_ri = pd.read_excel('g_trend.xlsx', sheet_name='etf_ri').set_index('ref_date').truncate(before=st).iloc[:, 0]

iots = iots.set_index('ref_date').truncate(before=st)
stocks_ris = stocks_ris.set_index('ref_date').truncate(before=st)

trend_strength_dict = {}
plot_stuff = False
fb = False
hw = True
to_iter = iots.loc[pd.to_datetime(st_roll):, :].index
pd.plotting.register_matplotlib_converters()

for i in ['US34959E1091']:  # iots.columns:

    for cnt, dt in enumerate(to_iter):
        isin = iots.loc[:, i].name
        stock_int = iots.loc[dt - pd.to_timedelta(365, 'D'):dt, i]
        if hw:
            data = stock_int
            # fit model
            model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=52)
            model_fit = model.fit()
            t_strength = model_fit.slope.values[-1]

        elif fb:
            ts_decomp = sm.tsa.seasonal.seasonal_decompose(stock_int, model='multiplicative')
            trend = ts_decomp.trend
            cycle, trend = smapi.tsa.filters.hpfilter(stock_int, 1600)

            # gdp_decomp = dta[['realgdp']]
            # gdp_decomp["cycle"] = cycle
            # gdp_decomp["trend"] = trend

            t_strength = get_trend_strength(stock_int, trend, fb=fb)

        else:
            # FB Prophet
            to_fit = stock_int.reset_index()
            to_fit.columns = ['ds', 'y']
            m = Prophet(seasonality_mode='multiplicative')
            m.fit(to_fit)
            future = m.make_future_dataframe(periods=10, freq='W')
            forecast = m.predict(future)
            stock_ri = stocks_ris.loc[:, i].dropna()
            er_ts = excess_return_calc(stock_ri, sp_ri)
            t_strength = get_trend_strength(stock_int, forecast, fb=fb)

        trend_strength_dict[dt] = t_strength
        print(cnt, len(to_iter))

        if plot_stuff:
            plot_int_er_trend(stock_int, er_ts, t_strength)
            fig1 = m.plot(forecast)
            breakp = add_changepoints_to_plot(fig1.gca(), m, forecast)
            fig2 = m.plot_components(forecast)
            plt.tight_layout()
            plt.show()

trends = pd.DataFrame(trend_strength_dict)
for i in trends.columns:
    trends.loc[i:, i] = np.nan

trends.plot(alpha=0.5, legend=False, linewidth=0.5, color='gray')
trends.apply(squeeze_nan, axis=1).iloc[:, 0].plot(color='black')
plt.show()
quit()
print(stocks_ris.resample('W-FR').first().shape)
print(pd.concat(trend_strength_dict, axis=1).shape)

for i in iots.columns:
    i = 'US2561631068'

    pd.plotting.register_matplotlib_converters()
    isin = iots.loc[:, i].name
    stock_int = iots.loc[:, i]

    # FB Prophet
    to_fit = stock_int.reset_index()
    to_fit.columns = ['ds', 'y']
    m = Prophet(seasonality_mode='multiplicative')
    m.fit(to_fit)
    future = m.make_future_dataframe(periods=10, freq='W')
    forecast = m.predict(future)

    if plot_stuff:
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)

        stock_ri = stocks_ris.loc[:, i].dropna()
        er_ts = excess_return_calc(stock_ri, sp_ri)
        stock_int.plot(ax=ax)
        er_ts.plot(ax=ax)

        fig1 = m.plot(forecast)
        breakp = add_changepoints_to_plot(fig1.gca(), m, forecast)
        fig2 = m.plot_components(forecast)
        plt.show()

    quit()
    for n in range(1, 20):
        print(n)
        up_to = (stock_int.dropna().index[0].normalize() + pd.to_timedelta(1, 'Y')) + pd.to_timedelta(n, 'W')
        to_fit = stock_int.truncate(after=up_to)
        to_fit = to_fit.reset_index()
        to_fit.columns = ['ds', 'y']
        m = Prophet(seasonality_mode='multiplicative')
        m.fit(to_fit, verbose=False)

        print('ciao')


    if plot_stuff:
        trend_strength.plot()
        plt.show()

print(pd.DataFrame(trend_strength_dict).T)

print('ciao')
