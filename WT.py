import numpy as np
import pandas as pd
import os.path as pth
import matplotlib.pyplot as plt

from WT_functions import backtest_strategy
from WT_functions import get_and_tidy_up_data
from WT_functions import single_commodity_optimal_contract


df, names_mapping = get_and_tidy_up_data(filename='contracts_prices.csv',
                                         filename_info='contracts_info.csv')
codes = df.contract_code.unique()

# initialize dictionaries
weights_dict = {}
price_df_dict = {}
contracts_df_dict = {}
weights_mask_dict = {}
chosen_contracts_dict = {}

for cd in df.contract_code.unique():
    print('Choose Optimal Contract for Single Commodity: {}'.format(cd))
    contracts_df_single, chosen_contracts_single, price_df_single_comm, weights_single, weights_single_mask = \
        single_commodity_optimal_contract(df[df.contract_code == cd])

    weights_dict[cd] = weights_single
    price_df_dict[cd] = price_df_single_comm
    contracts_df_dict[cd] = contracts_df_single
    weights_mask_dict[cd] = weights_single_mask
    chosen_contracts_dict[cd] = chosen_contracts_single

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
