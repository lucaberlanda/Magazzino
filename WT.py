import numpy as np
import pandas as pd
import os.path as pth
import matplotlib.pyplot as plt
from WT_functions import IndexConstruction
from WT_functions import plot_single_contract

filename = 'contracts_prices.csv'
filename_info = 'contracts_info.csv'
filepath = pth.join(pth.join('Other', 'WisdomTree'), filename)
filepath_info = pth.join(pth.join('Other', 'WisdomTree'), filename_info)
from time import time

df = pd.read_csv(filepath).set_index('date')
df_info = pd.read_csv(filepath_info).set_index('contract_code')
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
    aa = list(
        df_flt.reset_index().groupby(['date', 'contract_code'])['roll_yield'].idxmax().dropna().values.astype(int))

    chosen_contracts = df_flt.reset_index().iloc[aa].set_index('date')

    cols = ['contract_code', 'mat_month', 'mat_year']
    price_df_comm = df.set_index(cols, append=True).loc[:, 'close'].unstack(cols)
    # sort the columns properly
    price_df_comm.sort_index(level='mat_year', axis=1, inplace=True)

    w_df_mask = ~chosen_contracts.set_index(cols, append=True).loc[:, 'close'].unstack(cols).isnull()
    w_df = w_df_mask.astype(float)  # convert bool for chosen contract in a float

    # make sure that the weight matrix and price matrix have same indexes
    w_df = w_df.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)
    w_df_mask = w_df_mask.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)

    return price_df_comm, w_df, w_df_mask


def backtest_strategy(prices, w):
    w = w.copy()
    prices = prices.copy()

    w.columns = w.columns.droplevel(0)
    prices.columns = prices.columns.droplevel(0)

    # create single level column to pass the DataFrame
    w.columns = ['_'.join(map(str, x)) for x in w.columns]
    prices.columns = ['_'.join(map(str, x)) for x in prices.columns]

    weights_at_reb_dt = w.resample('BM').last().fillna(0).copy()
    idx_cls = IndexConstruction(prices, weights_at_reb_dt.T, name='strategy', rebalancing_f='weights')
    idx_cls.get_index()
    return idx_cls, idx_cls.idx


codes = df.contract_code.unique()

# initialize dictionaries
price_df_dict = {}
weights_dict = {}
weights_mask_dict = {}

for cd in df.contract_code.unique():
    print('Choose Optimal Contract for Single Commodity: {}'.format(cd))
    price_df_single_comm, \
    weights_single, \
    weights_single_mask = single_commodity_optimal_contract(df[df.contract_code == cd])

    price_df_dict[cd] = price_df_single_comm
    weights_dict[cd] = weights_single
    weights_mask_dict[cd] = weights_single_mask

price_df = pd.concat(price_df_dict, axis=1)
weights = pd.concat(weights_dict, axis=1)

weights = weights.shift().dropna(how='all')
price_df = price_df.truncate(before=weights.index[0])

ts_dict = {}
for cd in codes:
    print('Backtest Strategy For Single Commodity: {}'.format(cd))
    price_df_single = price_df.loc[:, cd]
    weights_single = weights.loc[:, cd]
    strategy_class, strategy_ts = backtest_strategy(prices=price_df_single, w=weights_single)
    ts_dict[cd] = strategy_ts

print('Backtest Full Strategy')
ptf_class, ptf_ts = backtest_strategy(prices=price_df, w=weights)
ts_dict['portfolio'] = ptf_ts

ptf_ts.plot()
plt.show()
# todo check sizes
# missing dates in w
print(pd.Series(list(set(price_df.index) - set(weights.index))).sort_values())


# check missing business days
pd.bdate_range(start = i[1].index[0], end = i[1].index[-1] ).difference(i[1].index)