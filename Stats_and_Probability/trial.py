import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
number_of_trials_dict = {}
for i in range(10000):
    uniform_rv = np.random.uniform(size=50)
    pass_one = pd.Series(uniform_rv.cumsum()) > 1
    number_of_trials = pass_one[pass_one == True].index.tolist()[0] + 1
    number_of_trials_dict[i] = number_of_trials

print(pd.Series(number_of_trials_dict).mean())

print('ciao')