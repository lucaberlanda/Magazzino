import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# PARAMETERS
trials = 1000
samples = 100000
prob_of_success = 0.5
percentage_distance_from_mean = 0.5
expected_value = prob_of_success * trials

##########

percentage_distance_from_mean_range = np.arange(0, 0.5, 0.01)
generic_range = range(0, 50, 1)

"""
Let X be a binomially distributed random variable with parameters n and p. As n gets large while np remains fixed, the
distribution of X is well approximated by the Poisson distribution with parameter λ = np.
"""

n = range(20, 100000, 10000)
n_p = 10  # starting from 10 * 20% (p=20%)

for i in n:
    p = n_p / i
    simulation = np.random.binomial(i, p, 10000)
    plt.hist(simulation, 50, facecolor='green', alpha=0.75)

plt.show()

def chernoff_bounds(percentage_distance_from_mean, expected_value):

    # the first chernoff bound is for the percentage_distance_from_mean > 0
    chernoff_bound_one = (np.exp(percentage_distance_from_mean) / (
        (1 + percentage_distance_from_mean) ** (1 + percentage_distance_from_mean))) ** expected_value

    # the first chernoff bound is for the 0 < percentage_distance_from_mean <= 1
    chernoff_bound_two = 2 * (np.exp((- expected_value * (percentage_distance_from_mean ** 2)) / 3))

    bounds = pd.Series([chernoff_bound_two, chernoff_bound_one])

    return bounds

# plot various chernoff bounds for various distance from mean __________________________________________________________

dict_to_store = {}

for i, j in zip(percentage_distance_from_mean_range, generic_range):
    dict_to_store[j] = chernoff_bounds(i, expected_value)

bounds_evolution = pd.DataFrame(dict_to_store).T
bounds_evolution.plot()
plt.show()


successes = pd.Series(np.random.binomial(trials, prob_of_success, samples))
distance_from_exp_value = (successes - expected_value) / expected_value
plt.hist(distance_from_exp_value, 50, facecolor='green', alpha=0.75)
plt.show()

quit()
