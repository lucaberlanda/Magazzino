import primitive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot


def plot_int_er_trend(int, exc_ret_ts, strength):

    fig = plt.figure(figsize=(10, 8), facecolor='white')
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    int.plot(ax=ax1, linestyle="", marker="o", color='#e85b6e', markersize=3)
    int.plot(ax=ax1, alpha=0.5, color='#e85b6e', linewidth=1)
    exc_ret_ts.plot(ax=ax2, color='#042c58')

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


def get_trend_strength(fcst):
    trend_strength = fcst.set_index('ds').trend.shift(-1) - fcst.set_index('ds').trend
    trend_strength.index = [pd.to_datetime(dt) for dt in trend_strength.index]
    return trend_strength


st = '01-01-2010'
iots = pd.read_excel('g_trend.xlsx', sheet_name='interest_over_time').truncate(before=st)
stocks_ris = pd.read_excel('g_trend.xlsx', sheet_name='stocks_ris').truncate(before=st)
sp_ri = pd.read_excel('g_trend.xlsx', sheet_name='S&P_ri').set_index('ref_date').truncate(before=st).iloc[:, 0]
etf_ri = pd.read_excel('g_trend.xlsx', sheet_name='etf_ri').set_index('ref_date').truncate(before=st).iloc[:, 0]
iots = iots.set_index('ref_date').truncate(before=st)
stocks_ris = stocks_ris.set_index('ref_date').truncate(before=st)
trend_strength_dict = {}
plot_stuff = False

for i in iots.columns:

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

    stock_ri = stocks_ris.loc[:, i].dropna()
    er_ts = primitive.excess_return_calc(stock_ri, sp_ri)
    t_strength = get_trend_strength(forecast)

    trend_strength_dict[i] = t_strength

    if plot_stuff:
        plot_int_er_trend(stock_int, er_ts, t_strength)
        fig1 = m.plot(forecast)
        breakp = add_changepoints_to_plot(fig1.gca(), m, forecast)
        fig2 = m.plot_components(forecast)
        plt.tight_layout()
        plt.show()

print(pd.concat(trend_strength_dict, axis=1).shape)
print(stocks_ris.resample('W-FR').first().shape)
quit()

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
        er_ts = primitive.excess_return_calc(stock_ri, sp_ri)
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
