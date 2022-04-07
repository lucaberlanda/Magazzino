import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Finance.Portfolio.stocks import Stock


def vol_term_structure(ri):
    std_dict = {}
    theoretical_std_dict = {}
    first_std = ri.pct_change().std()
    for i in range(1, 6000, 10):
        std_dict[i] = ri.pct_change(periods=i).std()
        theoretical_std_dict[i] = first_std * np.sqrt(i)

    fig1 = plt.figure(figsize=(6, 5))
    ax = fig1.add_subplot(1, 1, 1)
    pd.Series(std_dict, name='std').plot(logy=True, logx=True, marker='o', markersize=3, linewidth=0, color='b',
                                         alpha=0.3, ax=ax)
    pd.Series(theoretical_std_dict).plot(logy=True, logx=True, color='black', ax=ax, label=True, alpha=0.2)
    ax.set_title('Volatility Term Structure')
    plt.show()


risky_tk = 'SPY'
return_index = Stock(risky_tk).price().loc[:, 'Adj Close']
vol_term_structure(return_index)
