import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# AVMR >>> AVERAGE MONTHLY VOLATILITY RANKINGS
# Daily S&P500 data from 1986==>2018
url = "https://raw.githubusercontent.com/Patrick-David/AMVR/master/spx.csv"
df = pd.read_csv(url, index_col='date', parse_dates=True)
daily_ret = df['close'].pct_change().dropna()
original_index = daily_ret.index
# use pandas to resample returns per month and take standard deviation as measure of Volatility
# then annualize by multiplying by sqrt of number of periods (12)
mnthly_annu = daily_ret.resample('M').std() * np.sqrt(12)
print(mnthly_annu.head())

# we can see major market events show up in the volatility
plt.plot(mnthly_annu)
plt.axvspan('1987', '1989', color='r', alpha=.5)
plt.axvspan('2008', '2010', color='r', alpha=.5)
plt.title('Monthly Annualized vol — Black Monday and 2008 Financial Crisis highlighted')
labs = mpatches.Patch(color='red', alpha=.5, label="Black Monday & '08 Crash")
plt.legend(handles=[labs])
plt.show()

# for each year, rank each month based on volatility lowest=1 Highest=12
# average the ranks over all years for each month
ranked = mnthly_annu.groupby(mnthly_annu.index.year).rank()
final = ranked.groupby(ranked.index.month).mean()
final.describe()

# plot results for S&P AMVR: clearly October has the highest ave vol rank and December has the lowest.
# Mean of 6.45 is plotted

b_plot = plt.bar(final.index, final)
b_plot[9].set_color('g')
b_plot[11].set_color('r')

fin = abs(final - final.mean())
print(fin.sort_values())
Oct_value = fin[10]
Dec_value = fin[12]
print('Extreme Dec value:', Dec_value)
print('Extreme Oct value:', Oct_value)

# as our Null is that no seasonality exists or alternatively that the month / day does not matter in terms of AMVR,
# we can shuffle 'date' labels. for simplicity, we will shuffle the 'daily' return data
new_df_sim = pd.DataFrame()
highest_only = []
count = 0
n = 10

for i in range(n):
    print(i)
    # sample same size as dataset, drop timestamp
    daily_ret_shuffle = daily_ret.sample(len(daily_ret.index)).reset_index(drop=True)
    # add new timestamp to shuffled data
    daily_ret_shuffle.index = original_index

    # then follow same data wrangling as before...
    mnthly_annu = daily_ret_shuffle.resample('M').std() * np.sqrt(12)
    ranked = mnthly_annu.groupby(mnthly_annu.index.year).rank()
    sim_final = ranked.groupby(ranked.index.month).mean()

    # add each of 1000 sims into df
    new_df_sim = pd.concat([new_df_sim, sim_final], axis=1)

    # also record just highest AMVR for each year (we will use this later for p-hacking explanation)
    maxi_month = max(sim_final)
    highest_only.append(maxi_month)

# calculate absolute deviation in AMVR from the mean
all_months = new_df_sim.values.flatten()
mu_all_months = all_months.mean()
abs_all_months = abs(all_months - mu_all_months)
# calculate absolute deviation in highest only AMVR from the mean
mu_highest = np.mean(highest_only)
abs_highest = [abs(x - mu_all_months) for x in highest_only]

# count number of months in sim data where abs AMVR is > Dec; Here we are comparing against ALL months
count = 0
for i in abs_all_months:
    if i > Dec_value:
        count += 1
ans = count / len(abs_all_months)
print('p-value:', ans)

# calculate 5% significance
abs_all_months_95 = np.quantile(abs_all_months, .95)
abs_highest_95 = np.quantile(abs_highest, .95)

# plot the answer to Q1 in left column and Q2 in right column
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', figsize=(12, 12))

# plot 1
ax1.hist(abs_all_months, histtype='bar')
ax1.set_title('AMVR all months')
ax1.set_ylabel('Frequency')
n, bins, patches = ax3.hist(abs_all_months, density=1, histtype='bar', cumulative=True, bins=30)
ax3.set_ylabel('Cumulative probability')
ax1.axvline(Dec_value, color='b', label='Dec Result')
ax3.axvline(Dec_value, color='b')
ax3.axvline(abs_all_months_95, color='r', ls='--', label='5% Sig level')

# plot2
ax2.hist(abs_highest, histtype='bar')
ax2.set_title('AMVR highest only')
ax2.axvline(Dec_value, color='b')
n, bins, patches = ax4.hist(abs_highest, density=1, histtype='bar', cumulative=True, bins=30)
ax4.axvline(Dec_value, color='b')
ax4.axvline(abs_highest_95, color='r', ls='--')
ax1.legend()
ax3.legend()
