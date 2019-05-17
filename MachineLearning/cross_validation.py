from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# https://github.com/codebasics/py/blob/master/ML/12_KFold_Cross_Validation/12_k_fold.ipynb

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# KFold cross validation
# Basic example

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
KFold(n_splits=3, random_state=None, shuffle=False)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)

# Use KFold for our digits example

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

from sklearn.model_selection import cross_val_score
# Logistic regression model performance using cross_val_score

cross_val_score(LogisticRegression(), digits.data, digits.target)
# svm model performance using cross_val_score

# random forest performance using cross_val_score
cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target)

# cross_val_score uses stratified kfold by default
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5),digits.data, digits.target, cv=10)
