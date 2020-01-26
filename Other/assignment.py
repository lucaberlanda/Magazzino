# CREDIMI CHALLENGE
# Model for offering the best possible interest rate (i.e. maximizing the probability of being accepted) gived
# a total portfolio yield of 3.5%

import numpy as np
import collections
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

sns.set_style('white')
df_raw = pd.read_excel('futuro_applications_dataset.xlsx', sheet_name='dataset')
plot_int_vs_rating = False
get_spearman = False

aggregations = {'applicationDate': 'count', 'Interest Rate Proposed': 'median', 'fdgRating': 'mean'}
# todo funnel

# DATA CLEANING
df_raw.columns = df_raw.iloc[0, :]
df_raw = df_raw.drop(0)
cmp_list = df_raw.loc[:, 'Company ID'].values.tolist()
cmp_doubles = [item for item, count in collections.Counter(cmp_list).items() if count > 1]
df_raw = df_raw.set_index('Company ID')

for cmp in cmp_doubles:
    df_raw = df_raw.drop(cmp)

for i in df_raw.columns:
    df_raw = df_raw.replace({i: {'inf': np.nan}})

# dictionary to map credimi rating; since we will be using a Gradient Boosting /  Random Forest, we can overlook
# the non-linearity in the rating definition, hence assigning a range from 0 to 10
credimi_rating_mapping = {'A3': 1, 'A4': 2, 'B1': 3, 'B2': 4, 'B3': 5, 'B4': 6, 'C1': 7, 'C2': 8, 'C3': 9, 'C4': 10}
df_raw = df_raw.replace({'Credimi algoRating': credimi_rating_mapping})

# one hot encoding for Company Geo Area, ignore cities and region, prone to overfitting.
df_raw = pd.concat([df_raw, pd.get_dummies(df_raw.loc[:, 'Company Geo Area'])], axis=1)

y_mapping_dict_str = {'AcceptedByTheClient': 'Accepted',
                      'Financed': 'Accepted',
                      'ReadyForApproval': 'Accepted',
                      'Uninterested': 'Refused'}

y_mapping_dict = {'Accepted': 1,
                  'Refused': 0}

do_drop = ['Rejected']
df_approved = df_raw[df_raw.dossierStatus != 'Rejected']
df_tba = df_approved[df_approved.dossierStatus == 'ApprovedByCredimi']
df_answered = df_approved[df_approved.dossierStatus.isin(list(y_mapping_dict_str.keys()))]
df_answered = df_answered.replace({'dossierStatus': y_mapping_dict_str})

# Box Whisker Plot with Interest Rate ?
if plot_int_vs_rating:
    # PLOT INTEREST RATE VS RATING
    interest_vs_rating = df_answered.loc[:, ['Interest Rate Proposed', 'Credimi algoRating', 'dossierStatus']].dropna()
    fig = plt.figure(1, figsize=(11, 6))
    ax = fig.add_subplot(111)
    acc_labels = list(set(y_mapping_dict_str.values()))
    for acc_label in acc_labels:
        to_plot = interest_vs_rating[interest_vs_rating.dossierStatus == acc_label]
        ax.scatter(to_plot.loc[:, 'Interest Rate Proposed'], to_plot.loc[:, 'Credimi algoRating'], s=50, alpha=0.5)

    ax.legend(acc_labels)
    ax.set_xlabel('Interest Rate', size=13)
    ax.set_ylabel('Algo Rating', size=13)
    ax.set_title('Interest Rate vs Rating', size=20)
    plt.tight_layout()
    plt.show()

to_drop_cols = ['applicationDate', 'Credit Decision Outcome', 'approvedAt', 'ATECO Identifier', 'Company City',
                'Company Province', 'Company Geo Area', 'Company Region', 'Company ZIP Code', 'Juridicial Form',
                'ATECO', 'Credimi Industry', 'Company Start Date', 'cr_date_eoy', 'last_available_cr_date']

df_answered = df_answered.replace({'dossierStatus': y_mapping_dict}).drop(to_drop_cols, axis=1)
df_answered = df_answered.dropna(how='all', axis=1).fillna(df_answered.mean())

# seems reasonable to assume that the interest rate proposed would be lower than the one originally proposed
labels = df_answered.loc[:, 'dossierStatus']
features = df_answered.drop('dossierStatus', axis=1)
features_refusals = features.loc[labels[labels == 0].index, :]
features_labels = features.columns.tolist()

# compute SPEARMAN CORRELATION to get better how credit algo works
if get_spearman:
    spearman_dict = {}
    for i in features.columns:
        sp_corr = ss.spearmanr(pd.concat([features.loc[:, i], features.loc[:, 'Interest Rate Proposed']], axis=1))[0]
        print(sp_corr)
        spearman_dict[i] = sp_corr

    print(pd.Series(spearman_dict).sort_values(ascending=False))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)
# clf = RandomForestClassifier(n_estimators=500, max_depth=5)
clf = GradientBoostingClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)

feature_imp = pd.Series(clf.feature_importances_, index=features_labels).sort_values()
# pd.Series(clf.feature_importances, index=relevant_columns).sort_values(ascending=False).plot(kind='bar', color='blue')

print(accuracy_score(y_test, clf_predict))
print(confusion_matrix(y_test, clf_predict))

# trial with an item
new_comp_ptf = {}
for comp in features_refusals.index:
    comp_info = {}
    pred = {}
    ratio = {}
    old_acc_proba = 1
    original_int_rate = features_refusals.loc[comp, 'Interest Rate Proposed']
    for i in np.arange(1, 10, 0.5):
        features_refusals.loc[comp, 'Interest Rate Proposed'] = i
        acceptance_proba = clf.predict_proba(features_refusals.loc[comp, :].values.reshape(1, -1))[0][1]
        # make probability monotonous in order to avoid counter-intuitive behaviours
        if acceptance_proba > old_acc_proba:
            acceptance_proba = old_acc_proba

        pred[i] = acceptance_proba
        old_acc_proba = acceptance_proba

    pred_s = pd.Series(pred)
    proposed_int = max(pred_s[pred_s == max(pred_s)].index)
    comp_info['proposed_interest'] = proposed_int
    comp_info['acceptance_prob'] = round(max(pred_s), 3)
    comp_info['amount'] = features_refusals.loc[comp, 'amountProposed (€)']
    comp_info['old_interest'] = original_int_rate
    new_comp_ptf[comp] = comp_info

print('ciao')
quit()

for comp in features_refusals.index:
    pred = {}
    ratio = {}
    old_acc_proba = 1
    for i in np.arange(1, 10, 0.5):
        features_refusals.ix[comp, 'Interest Rate Proposed'] = i
        acceptance_proba = clf.predict_proba(features_refusals.iloc[item, :].values.reshape(1, -1))[0][1]
        # make probability monotonous in order to avoid counter-intuitive behaviours
        if acceptance_proba > old_acc_proba:
            acceptance_proba = old_acc_proba

        pred[i] = acceptance_proba
        ratio[i] = (1 - acceptance_proba) / i  # aka refusal probab (as low as possible) and yield /as high as possible)
        old_acc_proba = acceptance_proba

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    pd.Series(pred).plot(ax=ax1)
    pd.Series(ratio).plot(ax=ax2)
    plt.show()


pred_s = pd.Series(pred)
print(pred_s[pred_s == max(pred_s)])