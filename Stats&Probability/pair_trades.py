import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

long = 'twitter'
short = 'facebook'

xls = pd.ExcelFile('Shannon_investment_strategy.xlsx').parse(4)
xls.index = xls.loc[:, float('nan')]
xls = xls.drop(float('nan'), axis=1)
prices = xls.iloc[:, :2]
prices.columns = ['facebook', 'twitter']
returns = prices.pct_change()
prices_rebased = prices.apply(lambda x: x /x[0] * 100)

returns.loc[:, short] = returns.loc[:, short] * (-1)
strategy_ret = returns.sum(axis=1)
strategies_dict = {}
strategies_returns_dict = {}

for leverage in np.arange(0.5, 10, 0.01):
    strategy_returns_l = leverage * strategy_ret
    strategy_returns_l.iloc[0] = 0
    strategy_returns_l += 1
    strategy_equity_line = strategy_returns_l.cumprod() * 100
    strategy_return = (strategy_equity_line.values[-1] / strategy_equity_line.values[0])
    strategies_dict[leverage] = strategy_equity_line
    strategies_returns_dict[leverage] = strategy_return

strategy_returns = pd.Series(strategies_returns_dict)
strategy_returns.plot()
# strategies_df = pd.concat(strategies_dict, axis=1)
# strategies_df.plot()
plt.show()
