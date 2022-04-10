import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from WT_functions import Stats
from WT_functions import rebase_at_x
from WT_functions import plot_contract
from WT_functions import backtest_strategy
from WT_functions import get_and_tidy_up_data
from WT_functions import plot_future_curve_and_roll_yield
from WT_functions import single_commodity_optimal_contract

part_one = False
part_two = True
plot_stuff = False

df, names_mapping = get_and_tidy_up_data(filename='contracts_prices.csv',
                                         filename_info='contracts_info.csv')

if plot_stuff:
    fig1 = plot_contract('volume_USD', df, 'NG', [6, 7, 8, 9, 10, 11], [2020], names_mapping, return_fig=True)
    plt.savefig('volume_USD.png', transparent=True)

    fig2 = plot_contract('oi_USD', df, 'NG', [6, 7, 8, 9, 10, 11], [2020], names_mapping, return_fig=True)
    plt.savefig('oi_USD.png', transparent=True)

    fig3 = plot_contract('close', df, 'NG', [6, 7, 8, 9, 10, 11], [2020], names_mapping, return_fig=True)
    plt.savefig('close.png', transparent=True)

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

        if cd == 'NG' and plot_stuff:
            fig4 = plot_future_curve_and_roll_yield(contracts_df_single.loc[pd.to_datetime('2020-04-30')],
                                                    return_fig=True)
            plt.savefig('curve_and_roll_yield.png', transparent=True)

        weights_dict[cd] = weights_single
        price_df_dict[cd] = price_df_single_comm
        contracts_df_dict[cd] = contracts_df_single
        chosen_contracts_dict[cd] = chosen_contracts_single

    price_df = pd.concat(price_df_dict, axis=1)
    weights = pd.concat(weights_dict, axis=1)

    weights = weights.shift().dropna(how='all')
    price_df = price_df.truncate(before=weights.index[0])

    # Send data to pickle
    weights.to_pickle('weights')
    price_df.to_pickle('price_df')

else:
    weights = pd.read_pickle('weights')
    price_df = pd.read_pickle('price_df')

if part_two:
    codes = df.contract_code.unique()
    empty_contracts = weights.sum()[weights.sum() == 0].index
    price_df = price_df.drop(empty_contracts, axis=1)
    weights = weights.drop(empty_contracts, axis=1)

    ts_dict = {}

    #for cd in codes:
    #    print('Backtest Strategy For Single Commodity: {}'.format(cd))
    #    price_df_single = price_df.loc[:, cd]
    #    weights_single = weights.loc[:, cd]
    #    strategy_class, strategy_ts = backtest_strategy(prices=price_df_single,
    #                                                    w=weights_single,
    #                                                    rebalancing='monthly')
    #    ts_dict[cd] = strategy_ts

    print('Backtest Full Strategy')
    ptf_class, ptf_ts = backtest_strategy(prices=price_df, w=weights, rebalancing='monthly')
    ts_dict['portfolio'] = ptf_ts

    full_df = rebase_at_x(pd.concat(ts_dict, axis=1))

    # Send to excel
    full_df.to_excel('backtest_levels.xlsx')

else:
    full_df = pd.read_excel('backtest_levels.xlsx')

quit()
#  Stats computation and export
stats = Stats(full_df.pct_change())
summary_stats = stats.summary_stats()
summary_stats.to_csv('stats.csv')

# plot strategy
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
full_df.loc[:, 'portfolio'].plot(color='black', ax=ax, legend=True)
full_df.drop('portfolio', axis=1).rename(names_mapping, axis=1).plot(cmap='brg', ax=ax, alpha=0.5, linewidth=1)
ax.set_title('Portfolio vs. Single Commodities', fontsize=20)
plt.savefig('backtest_levels.png', transparent=True)

# Maximum Weight Plot
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(1, 1, 1)
ptf_class.w_df.max(axis=1).iloc[:1000].plot(linewidth=1, color='black')
ax.set_title('Maximum Weight Evolution', fontsize=15)
plt.savefig('maximum_weight_evolution.png', transparent=True)
