import numpy as np
import pandas as pd
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

X = df.drop('sepal length (cm)', axis=1)
y = df.loc[:, 'sepal length (cm)']

models_dict = {'SVR': SVR(C=1.0, epsilon=0.2),
               'RF': RandomForestRegressor(),
               'Ridge': Ridge(),
               'Adaboost': AdaBoostRegressor(),
               'Passive Aggressive': PassiveAggressiveRegressor()}

mean_score_l = {}
for size in np.arange(0.1, 0.9, 0.01):
    print(size)
    regr = make_pipeline(StandardScaler(), RandomForestRegressor())
    regr.fit(X, y)
    cv = ShuffleSplit(n_splits=int(1/size), test_size=size)
    mean_score_l[size] = np.mean(cross_val_score(regr, X, y, cv=cv, scoring='r2'))

pd.Series(mean_score_l).plot()
plt.show()