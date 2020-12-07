import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


"""
ERGODIC PROPERTY (equality of averages)
The expectation value of the observable is a constant (independent of time), and the finite-time average of the
observable converges to this constant with probability one as the averaging time tends to infinity.

Whether an observable possesses this property is crucial when assessing the significance of its expectation value

"""

sns.set_style('white')

# random gaussian
s = pd.Series(np.random.randn(10000))

s_roll_mean = s.expanding().mean()
s_roll_mean.plot()
plt.show()

s_cum_sum = s.cumsum()
s_cum_sum.plot()
plt.show()

s_cum_sum.expanding().mean().plot()
plt.show()
