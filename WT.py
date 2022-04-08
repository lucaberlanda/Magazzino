import numpy as np
import pandas as pd
import os.path as pth
import matplotlib.pyplot as plt
from WT_plots import plot_single_contract

filename = 'contracts_prices_one_comm.csv'
filename_info = 'contracts_info.csv'
filepath = pth.join(pth.join('Other', 'WisdomTree'), filename)
filepath_info = pth.join(pth.join('Other', 'WisdomTree'), filename_info)

df = pd.read_csv(filepath).set_index('date')
df_info = pd.read_csv(filepath_info).set_index('contract_code')
df.index = [pd.to_datetime(i, format='%d/%m/%Y') for i in df.index]
df['last_trade_date'] = pd.to_datetime(df['last_trade_date'], format='%d/%m/%Y')

df = df.reset_index().sort_values(['index', 'mat_year', 'mat_month'],
                                  ascending=[True, False, False]).set_index('index')

df = df.merge(df_info, left_on='contract_code', right_index=True)


def single_commodity_optimal_contract(df, volume_threshold=1e8, oi_threshold=3e7):
    # we define the value of the contract by multiplying the current price by the size of the contract
    df['value_USD'] = df.close.mul(df.contract_size)
    df['oi_USD'] = df.oi.mul(df.value_USD)
    df['volume_USD'] = df.volume.mul(df.value_USD)

    # todo n_months 12
    df_grouped = df.reset_index().groupby(['index', 'contract_code'])
    roll_yield = df_grouped['close'].pct_change().shift(-1) / 12
    roll_yield.index = df.index
    df['roll_yield'] = roll_yield

    df_flt = df[(df.volume_USD > volume_threshold) & (df.oi_USD > oi_threshold)]
    aa = list(
        df_flt.reset_index().groupby(['index', 'contract_code'])['roll_yield'].idxmax().dropna().values.astype(int))

    chosen_contracts = df_flt.reset_index().iloc[aa].set_index('index')

    print(chosen_contracts)

    cols = ['contract_code', 'mat_month', 'mat_year']
    price_df_comm = df.set_index(cols, append=True).loc[:, 'close'].unstack(cols)

    w_df_mask = ~chosen_contracts.set_index(cols, append=True).loc[:, 'close'].unstack(cols).isnull()
    w_df = w_df_mask.astype(float)  # convert bool for chosen contract in a float
    # make sure that the weight matrix and price matrix have same indexes
    w_df = w_df.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)
    w_df_mask = w_df_mask.reindex(price_df_comm.index).reindex(price_df_comm.columns, axis=1)

    # todo weights proportional to roll yield


print(df.contract_code.unique())
for cd in df.contract_code.unique():
    single_commodity_optimal_contract(df[df.contract_code == cd])
