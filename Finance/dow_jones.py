from primitive import *
import statsmodels.api as sm
import numpy as np

df = pd.read_csv('data_House_Prices_and_Crime_1.csv')
print(df[df.loc[:, 'index_nsa'] == df.index_nsa.median()])

Y = df.loc[:, 'index_nsa']
X = df.loc[:, ['Homicides', 'Assaults', 'Robberies']]
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()
print(results.params)


def plot_kurtosis(kurt_original, kurt_reshuffled):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    kurt_reshuffled.name = 'reshuffled data'
    kurt_original.name = 'original data'
    kurt_original.plot(kind='bar', ax=ax, alpha=0.5, legend=True)
    kurt_reshuffled.plot(ax=ax, style='--', alpha=0.5, color='red', legend=True)
    ax.hlines(3, ax.get_xticks().min(), ax.get_xticks().max(), linestyle='--', color='black')
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 5) != 0:
            t.set_visible(False)

    plt.tight_layout()
    plt.show()


index_name = 'dow_jones'
df = get_dow_jones()
dj = df.loc[:, index_name]
dj_ret = dj.pct_change()
dj_log_ret = np.log(dj) - np.log(dj.shift(1))

dj_ret = pd.Series(np.random.normal(dj_ret.mean(), dj_ret.std(), size=len(dj_ret.index)))
dj_ret.name = index_name
sorted_returns = dj_ret.dropna().sort_values(ascending=False).reset_index().loc[:, index_name]
negative_ret = sorted_returns[sorted_returns < 0]

K_dict = {}
negative_ret_flt = negative_ret[negative_ret < -0.02]
for i in negative_ret_flt.index:
    K_dict[-negative_ret_flt.loc[i]] = (negative_ret_flt.loc[i:] * -1).mean() / (-negative_ret_flt.loc[i])

pd.Series(K_dict).iloc[:-1].plot()
plt.show()

log_log_plot_with_threshold(dj_log_ret, threshold=0.01)

kurtosis_original = pd.Series(lagging_kurtosis(dj.reset_index().loc[:, 'dow_jones']))
kurtosis_dict_reshuffled = {}
for k in range(10):
    dj_reshuffled = reshuffled_ts(dj)
    kurtosis_dict_reshuffled[k] = lagging_kurtosis(dj_reshuffled)

kurtosis_reshuffled = pd.DataFrame(kurtosis_dict_reshuffled).mean(axis=1)
plot_kurtosis(kurtosis_original, kurtosis_reshuffled)

dj.pct_change().dropna().expanding().mean().plot()
dj.pct_change().dropna().expanding().std().plot()
dj.pct_change().dropna().expanding().skew().plot()
dj.pct_change().dropna().expanding().kurt().plot()

