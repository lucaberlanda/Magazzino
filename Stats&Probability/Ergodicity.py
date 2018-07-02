import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

"""

Definition of Ergodicity (Peters and Gell-Mann): 

The expectation value of the observable is a constant 
(independent of time), and the finite-time average of the observable converges to this constant with probability one 
as the averaging time tends to infinity

The two key quantities in our treatment are the expected rate of change in wealth and the expected exponential growth 
rate of wealth

The problem they tried to solve is to overcome the expected value issues without referring trìo human psychology. 

***

Case 1: ADDITIVE REPETITION

***

Case 2: MULTIPLICATIVE REPETITION

"""

# PARAMETERS
trials = 1000
tail_mul = 1.5
head_mul = 0.6
tail_add = 0.5
head_add = -0.4

###

percentage_distance_from_mean_range = np.arange(0, 0.5, 0.01)
generic_range = range(0, 50, 1)
np.random.seed(0)

simulation1 = pd.Series(np.random.choice([tail_add, head_add], trials))
simulation2 = pd.Series(np.random.choice([tail_mul, head_mul], trials))

simulation1.iloc[0] = 1
simulation2.iloc[0] = 1

simulation1.cumsum().plot()
plt.show()
fig, ax = plt.subplots()
ax.plot(simulation2.cumprod())
ax.set_yscale('log')
plt.show()

print(simulation)
quit()
