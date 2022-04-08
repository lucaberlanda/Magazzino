import numpy as np
import pandas as pd
import os.path as pth
import matplotlib.pyplot as plt
from WT_plots import plot_single_contract
from logging import log

filename = 'contracts_prices_one_comm.csv'
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
df['oi_USD'] = df.oi.mul(df.value_USD)
df['volume_USD'] = df.volume.mul(df.value_USD)

lbls = ['contract_code', 'mat_month', 'mat_year']
# calculate the 22 days moving average (business days to get one full month moving average) and append as a column
volume_ma = df.groupby(lbls)['volume_USD'].rolling(window=22).mean()
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

    w_df_mask = ~chosen_contracts.set_index(cols, append=True).loc[:, 'close'].unstack(cols).isnull()
    w_df = w_df_mask.astype(float)  # convert bool for chosen contract in a float

    # make sure that the weight matrix and price matrix have same indexes
    w_df = w_df.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)
    w_df_mask = w_df_mask.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)

    return price_df_comm, w_df, w_df_mask


print(df.contract_code.unique())
for cd in df.contract_code.unique():
    log(1, cd)

    single_commodity_optimal_contract(df[df.contract_code == cd])
