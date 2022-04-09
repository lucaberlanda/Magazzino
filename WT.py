import numpy as np
import pandas as pd
import os.path as pth
import matplotlib.pyplot as plt

from WT_functions import rebase_at_x
from WT_functions import backtest_strategy
from WT_functions import get_and_tidy_up_data
from WT_functions import single_commodity_optimal_contract

part_one = False

df, names_mapping = get_and_tidy_up_data(filename='contracts_prices.csv',
                                         filename_info='contracts_info.csv')

if part_one:
    # initialize dictionaries
    weights_dict = {}
    price_df_dict = {}
    contracts_df_dict = {}
    chosen_contracts_dict = {}

    for cd in df.contract_code.unique():
        print('Choose Optimal Contract for Single Commodity: {}'.format(cd))
        contracts_df_single, chosen_contracts_single, price_df_single_comm, weights_single = \
            single_commodity_optimal_contract(df[df.contract_code == cd])

        weights_dict[cd] = weights_single
        price_df_dict[cd] = price_df_single_comm
        contracts_df_dict[cd] = contracts_df_single
        chosen_contracts_dict[cd] = chosen_contracts_single

    price_df = pd.concat(price_df_dict, axis=1)
    weights = pd.concat(weights_dict, axis=1)

    weights = weights.shift().dropna(how='all')
    price_df = price_df.truncate(before=weights.index[0])

    # writer = pd.ExcelWriter('weights.xlsx', engine='xlsxwriter')
    # writer2 = pd.ExcelWriter('price_df.xlsx', engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    weights.to_pickle('weights')
    price_df.to_pickle('price_df')

else:
    weights = pd.read_pickle('weights')
    price_df = pd.read_pickle('price_df')

codes = df.contract_code.unique()
ts_dict = {}
for cd in codes:
   print('Backtest Strategy For Single Commodity: {}'.format(cd))
   price_df_single = price_df.loc[:, cd]
   weights_single = weights.loc[:, cd]
   strategy_class, strategy_ts = backtest_strategy(prices=price_df_single, w=weights_single)
   ts_dict[cd] = strategy_ts

print('Backtest Full Strategy')
ptf_class, ptf_ts = backtest_strategy(prices=price_df, w=weights, )
ts_dict['portfolio'] = ptf_ts

empty_contracts = weights.sum()[weights.sum() == 0].index
price_df = price_df.drop(empty_contracts, axis=1)
weights = weights.drop(empty_contracts, axis=1)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)

to_plot = rebase_at_x(pd.concat(ts_dict, axis=1))

to_plot.loc[:, 'portfolio'].plot(color='black', ax=ax, legend=True)
to_plot.drop('portfolio', axis=1).rename(names_mapping, axis=1).plot(cmap='brg', ax=ax, alpha=0.5, linewidth=1)
ax.set_title('Portfolio vs. Single Commodities', fontsize=20)

plt.show()

ts_dict['portfolio'] = ptf_ts
