import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import CoreFunctions.functions_involved as finv
import seaborn as sns
import statsmodels as sm

sns.set_style('white')


def plot_acf(s):
    sm.graphics.tsaplots.plot_acf(s, lags=50)
    sm.graphics.tsaplots.plot_pacf(s, lags=50)
    plt.show()


time_series_df = finv.download_time_series_and_put_in_df([32718, ]).truncate(before='2010-12-30',
                                                                             after=finv.get_last_friday()). \
    pct_change().dropna()

plot_acf(time_series_df.iloc[:, 0])

