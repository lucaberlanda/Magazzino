from primitive import *
from TimeSeriesAnalysis.fb_prophet import RollingDecompose

iots = pd.read_excel('g_trend.xlsx', sheet_name='interest_over_time').set_index('ref_date')
ris = pd.read_excel('g_trend.xlsx', sheet_name='stocks_ris').set_index('ref_date')
sp_ri = pd.read_excel('g_trend.xlsx', sheet_name='S&P_ri').set_index('ref_date').iloc[:, 0]
stock_trends = {}

for ISIN in iots.columns:
    print(ISIN)
    # ISIN = 'US2561631068'
    iot = iots.loc[:, ISIN]
    ri = ris.loc[:, ISIN]
    decomposed = RollingDecompose(iot, method='prophet', last_date_only=True)
    stock_trends[ISIN] = decomposed.trend_strength

er_ts = excess_return_calc(ri, sp_ri)
decomposed.plot_int_er_trend(er_ts)


"""
first_surprise = surprise_dict[list(surprise_dict.keys())[0]]
if len(surprise_dict) == 1:
    surprises = first_surprise
else:
    last_surprises = {your_key: surprise_dict[your_key] for your_key in list(surprise_dict.keys())[1:]}
    surprises = pd.concat([first_surprise, pd.concat(last_surprises, axis=1).T], axis=0)

er_ts = excess_return_calc(stock_ri, sp_ri)

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
"""

# etf_ri = pd.read_excel('g_trend.xlsx', sheet_name='etf_ri').set_index('ref_date').iloc[:, 0]