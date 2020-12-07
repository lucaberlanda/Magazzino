# GRE CHALLENGE

import numpy as np
import pandas as pd
import seaborn as sb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.linear_model import Ridge, PassiveAggressiveRegressor

sns.set_style('white')
iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# C_mat = df.corr()
# print(C_mat)
# fig = plt.figure(figsize=(15, 15))
# sb.heatmap(C_mat, vmax=.8, square=True)
# plt.show()

X = df.drop('sepal length (cm)', axis=1)
y = df.loc[:, 'sepal length (cm)']

models_dict = {'SVR': SVR(C=1.0, epsilon=0.2),
               'RF': RandomForestRegressor(),
               'Ridge': Ridge(),
               'Adaboost': AdaBoostRegressor(),
               'Passive Aggressive': PassiveAggressiveRegressor()}

regr_l = {}
for model in models_dict.keys():
    regr = make_pipeline(StandardScaler(), models_dict[model])
    regr.fit(X, y)
    cv = ShuffleSplit(n_splits=1, test_size=0.80)
    print(model, cross_val_score(regr, X, y, cv=cv, scoring='r2'))
    print(model, regr.score(X, y))
    regr_l[model] = regr

for regr in regr_l.keys():
    fig = plt.figure(1, figsize=(11, 6))
    ax = fig.add_subplot(111)
    pred_df = pd.concat([pd.Series(regr_l[regr].predict(X)), y], axis=1)
    pred_df.columns = ['sepal_length_pred', 'sepal_length']
    pred_df = pred_df.sort_values('sepal_length').reset_index().drop('index', axis=1)
    pred_df.reset_index().plot(kind='scatter', x='index', y='sepal_length', color='black', ax=ax)
    pred_df.reset_index().plot(kind='scatter', x='index', y='sepal_length_pred', color='blue', ax=ax)
    ax.set_title(regr, fontsize=15)
    plt.show()
