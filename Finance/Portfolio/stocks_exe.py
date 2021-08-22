import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import time
from primitive import rebase_at_x
from scipy.optimize import minimize
from Finance.Portfolio.stocks import Stock

risky_tk = '^SP500TR'
risk_free_tk = '^FVX'


def total_return(leverage, return_indexes):
    w = pd.Series([leverage[0], 1 - leverage[0]], index=['risky', 'risk_free'])
    equity_line = (return_indexes.dot(w) + 1).cumprod()
    ret = equity_line.values[-1] / equity_line.values[-0] - 1
    ret_to_optimize = -ret
    return ret_to_optimize


def expanding_optimal_leverage(return_indexes):
    leverage_initial_guess = 1
    optimal_leverage_dict = {}
    for dt in ris.index[100:]:
        print(dt)
        optimal_leverage = minimize(total_return, [leverage_initial_guess], args=return_indexes.truncate(after=dt)).x[0]
        optimal_leverage_dict[dt] = optimal_leverage

    return optimal_leverage_dict


def leveraged_returns(leverage, return_indexes):
    """

    :param leverage: float; leverage of the risky asset
    :param return_indexes:
    :return:

    """

    w = pd.Series([leverage, 1 - leverage], index=['risky', 'risk_free'])
    equity_line = (return_indexes.dot(w) + 1).cumprod()
    ret = equity_line.values[-1] / equity_line.values[-0] - 1
    return equity_line, ret


def time_average_return_plot():
    l_range = np.arange(-10, 15, 0.25)
    annualized_g_dict = {}
    for l in l_range:
        ri, ret = leveraged_returns(l, ris)
        annualized_g = np.log(ret + 1) / ((ri.index[-1] - ri.index[0]).days / 365)
        annualized_g_dict[l] = annualized_g

    pd.Series(annualized_g_dict).plot()
    plt.show()


risky_ri = Stock(risky_tk).price().loc[:, 'Adj Close']
risky_ret = risky_ri.pct_change()
risk_free_yield = Stock(risk_free_tk).price().loc[:, 'Adj Close']
risk_free_yield_d = risk_free_yield / (365 * 100)

ris = risky_ret.to_frame().join(risk_free_yield_d, rsuffix=' risk free').ffill().dropna()
ris.columns = ['risky', 'risk_free']
l_range = np.arange(-4, 9, 0.25)
annualized_g_dict = {}
for l in l_range:
    ri, ret = leveraged_returns(l, ris)
    annualized_g = np.log(ret) / ((ri.index[-1] - ri.index[0]).days / 365)
    annualized_g_dict[l] = annualized_g

ret_dict = {}

mu_rf = ris.risk_free.mean()
mu_s = ris.risky.mean()
mu_e = mu_s - mu_rf
sigma_s = ris.risky.std()

g_l_dict = {}
l_range = np.arange(-3, 9, 0.1)
for l in l_range:
    g_l = mu_rf + l * mu_e - ((l ** 2 * sigma_s ** 2) / 2)
    g_l_dict[l] = g_l

mu_rf = ris.risk_free.mean()
mu_s = ris.risky.mean()
mu_e = mu_s - mu_rf
sigma_s = ris.risky.std()

g_l_dict = {}
l_range = np.arange(-3, 9, 0.1)
for l in l_range:
    g_l = mu_rf + l * mu_e - ((l ** 2 * sigma_s ** 2) / 2)
    g_l_dict[l] = g_l

mu_e = mu_s - mu_rf

for l in l_range:
    print(-total_return([l], ris))

"""
for l in l_range:
    ptf = rebase_at_x((ris.dot(w) + 1).cumprod())
    ret_dict[l] = ptf.values[-1] / ptf.values[-0] - 1

pd.Series(ret_dict).plot()
plt.show()
"""
