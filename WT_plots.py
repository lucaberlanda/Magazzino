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
