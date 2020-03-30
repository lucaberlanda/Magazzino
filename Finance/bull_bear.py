from primitive import *

import numpy as np
import matplotlib.pyplot as plt

instr_id = 22
market = 'Dow Jones'


def get_first_switch(df1):
    rolling_ret = (df1 / df1.iloc[0, 0] - 1).values.flatten()
    for idx_ret in rolling_ret:
        if idx_ret < -0.2:
            return 'bear'
        elif idx_ret > 0.2:
            return 'bull'


ri = pd.read_excel('djindus.xlsx', sheet_name='RI')
pi = pd.read_excel('djindus.xlsx', sheet_name='PI')
ri = ri.rename(columns={'Name': 'ref_date'})
ri = ri.set_index('ref_date')
ri.columns = [instr_id]
pi = pi.rename(columns={'Name': 'ref_date'})
pi = pi.set_index('ref_date')
pi.columns = [instr_id]
df = pd.concat([pi.loc[:ri.index[0], :].iloc[:-1, :], ri])

first_mkt = get_first_switch(df)
switch = first_mkt
reference_price = df.iloc[0, 0]
switch_dates = [df.index[0]]
bull_bear_dates = [df.index[0]]

for ref_date in df.index:
    current_price = df.loc[ref_date, instr_id]

    if switch == 'bear':
        reference_price = min(df.loc[switch_dates[-1]:ref_date, instr_id])
        reference_ret = current_price / reference_price - 1

    else:  # switch == 'bull'
        reference_ret = current_price / reference_price - 1
        reference_price = max(df.loc[switch_dates[-1]:ref_date, instr_id])

    if abs(reference_ret) > 0.2:

        if reference_ret > 0.2:
            switch = 'bull'
            bull_bear_dates.append(df.loc[switch_dates[-1]:ref_date, instr_id].idxmin())
        else:
            switch = 'bear'
            bull_bear_dates.append(df.loc[switch_dates[-1]:ref_date, instr_id].idxmax())

        switch_dates.append(ref_date)
        reference_price = current_price

switch_dates.append(df.index[-1])
bull_bear_dates.append(df.index[-1])

zero_line = 0  # zero for zero line at 100 or -100 for zero line at zero
n_bull_bear = np.arange(len(bull_bear_dates) - 1)
fig1 = plt.figure(figsize=(15, 10))
fig2 = plt.figure(figsize=(15, 10))
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = fig2.add_subplot(1, 1, 1)
color_to_choose = {-1: 'red', 1: 'green'}

color_n = -1
if first_mkt == 'bull':
    color_n = 1

rebased_series_dict = {}
for i in n_bull_bear:
    print(df.loc[bull_bear_dates[i]:bull_bear_dates[i+1], instr_id])
    to_plot = rebase_at_x(df.loc[bull_bear_dates[i]:bull_bear_dates[i+1], instr_id]) + zero_line
    print(to_plot)
    rebased_series_dict[i] = to_plot
    to_plot.plot(ax=ax1, color=color_to_choose[color_n])
    ax1.fill_between(to_plot.index, to_plot, zero_line + 100, facecolor=color_to_choose[color_n], alpha=0.4)
    to_plot.plot(ax=ax2, color=color_to_choose[color_n], logy=True)
    ax2.fill_between(to_plot.index, to_plot, zero_line + 100, facecolor=color_to_choose[color_n], alpha=0.4)
    color_n = color_n * -1

ax1.set_title('%s - Bull and Bear Markets' % market, fontsize=20)
ax2.set_title('%s - Bull and Bear Markets (log scale)' % market, fontsize=20)
fig1.savefig('bull_bear_%s.png' % market, transparent=False)
fig2.savefig('bull_bear_%s_log.png' % market, transparent=False)

summary_table = pd.concat([pd.Series(bull_bear_dates).shift(1), pd.Series(bull_bear_dates)], axis=1).dropna()
summary_table.columns = ['start', 'end']

bull_bear_list = []
market_status = {-1: 'bull', 1: 'bear'}
market_switch = 1
if first_mkt == 'bull':
    market_switch = -1

for i in summary_table.index:
    bull_bear_list.append(market_status[market_switch])
    market_switch *= -1

xiod_ret = []
xiod_dur = []
for i in summary_table.index:
    xiod_start = summary_table.loc[i, 'start']
    xiod_end = summary_table.loc[i, 'end']
    xiod_ri = df.loc[xiod_start:xiod_end, instr_id]
    ret = xiod_ri.iloc[-1] / xiod_ri.iloc[0] - 1
    dur = (xiod_end - xiod_start).days / 365
    xiod_ret.append(ret)
    xiod_dur.append(dur)

summary_table['period_return'] = xiod_ret
summary_table['period_duration_years'] = xiod_dur

rebased_df_to_excel = pd.concat(rebased_series_dict, axis=1)
writer = pd.ExcelWriter('bull_bear_%s.xlsx' % market, engine='xlsxwriter')
rebased_df_to_excel.to_excel(writer, sheet_name='ri')
summary_table.to_excel(writer, sheet_name='summary')
writer.save()