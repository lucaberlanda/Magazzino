# CREDIMI CHALLENGE
# Model for offering the best possible interest rate (i.e. maximizing the probability of being accepted) gived
# a total portfolio yield of 3.5%

import numpy as np
import collections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

sns.set_style('white')
df_raw = pd.read_excel('futuro_applications_dataset.xlsx', sheet_name='dataset')
plot_int_vs_rating = False

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

df_answered = df_answered.replace({'dossierStatus': y_mapping_dict})

df_raw.loc[:, 'Interest Rate Proposed'] = df_raw.loc[:, 'Interest Rate Proposed']. \
    fillna(df_raw.loc[:, 'Interest Rate Proposed'].median())

df_raw.loc[:, 'fdgRating'] = df_raw.loc[:, 'fdgRating'].fillna(df_raw.loc[:, 'fdgRating'].mean())
aggregations = {'applicationDate': 'count', 'Interest Rate Proposed': 'median', 'fdgRating': 'mean'}

# ptf without the rejected / not eligible companies
# the acceptance depend on a variable that you should change

# seems reasonable to assume that the interest rate proposed would be lower than the one originally proposed
y_mapping_dict = {'AcceptedByTheClient': 1,
                  'Financed': 1,
                  'ReadyForApproval': 1,
                  'Uninterested': 0}

relevant_columns = ["requestedAmount (€)", "Interest Rate Proposed",
                    "fdgAvailablePlafond (€)", "fdgRating", "Fatturato (€'000)", "Ebitda Margin (%)",
                    "Tasso Medio Pagato (Stima da Bilancio, %)", "Tasso Medio Pagato (Stima da Centrale Rischi, %)",
                    "cerved_structural_score_c6gvrn", "cash (€'000)", "consolidatedLiabilities (€'000)",
                    "utilizzo_scadenza_t-2 (€'000)", "utilizzo_scadenza_t-3 (€'000)", "utilizzo_scadenza_t-4 (€'000)",
                    "utilizzo_scadenza_t-5 (€'000)", "last_tension_autoliquidanti",
                    "last_tension_scadenza", "avg_6m_tension_autoliquidanti", "avg_6m_tension_revoca",
                    "avg_6m_tension_scadenza", "Age", "accordato_revoca_t-4 (€'000)", "accordato_revoca_t-5 (€'000)",
                    "accordato_scadenza_t (€'000)", "accordato_scadenza_t-1 (€'000)", "accordato_scadenza_t-2 (€'000)",
                    "accordato_scadenza_t-3 (€'000)", "accordato_scadenza_t-4 (€'000)",
                    "accordato_scadenza_t-5 (€'000)", "affidanti_t", "affidanti_t-1", "affidanti_t-2", "affidanti_t-3",
                    "affidanti_t-4", "affidanti_t-5", "last_accordato_tot"]

labels = df_answered.dossierStatus
features = df_answered.loc[:, relevant_columns]
features = features.fillna(features.median())

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4)
# clf = RandomForestClassifier(n_estimators=500, max_depth=5)
clf = GradientBoostingClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)

feature_imp = pd.Series(clf.feature_importances_, index=relevant_columns).sort_values()

# pd.Series(clf.feature_importances_, index=relevant_columns).\
# sort_values(ascending=False).plot(kind='bar', color='blue')

print(accuracy_score(y_test, clf_predict))
print(confusion_matrix(y_test, clf_predict))
print(clf.feature_importances_)

print(pd.DataFrame(clf.predict_proba(features)).mean())

features.loc[:, "Interest Rate Proposed"] = features.loc[:, "Interest Rate Proposed"] - \
                                            features.loc[:, "Interest Rate Proposed"] * 0.5

print(pd.DataFrame(clf.predict_proba(features)).mean())

# Given the data at hand, develop a pricing model that, for each company, computes the optimal
# price,
# I.E. THE INTEREST RATE THAT MAXIMIZES THE PROBABILITY OF THAT CLIENT ACCEPTING THE OFFER.

import scipy.stats as ss

for i in df_answered.columns:
    try:
        print(i, ss.spearmanr(df_answered.loc[:, ['Interest Rate Proposed', i]]))
    except:
        print(i, 'error')

# Relation between probability of accepting and yield
# Maximize this ratio

# trial with an item
items = np.arange(0, 500, 10)
for item in items:
    pred = {}
    ratio = {}
    ratio2 = {}
    old_acc_proba = 1
    for i in np.arange(1, 10, 0.5):
        features.ix[item, 'Interest Rate Proposed'] = i
        acceptance_proba = clf.predict_proba(features.iloc[item, :].values.reshape(1, -1))[0][1]

        if acceptance_proba > old_acc_proba:  # make it monotonous
            acceptance_proba = old_acc_proba

        pred[i] = acceptance_proba
        ratio[i] = (1 - acceptance_proba) / i  # aka refusal probab (as low as possible) and yield /as high as possible)
        ratio2[i] = acceptance_proba * 10 + i  # aka refusal probab (as low as possible) and yield /as high as possible)
        old_acc_proba = acceptance_proba

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    pd.Series(pred).plot(ax=ax1)
    pd.Series(ratio).plot(ax=ax2)
    pd.Series(ratio2).plot(ax=ax3)
    plt.show()
