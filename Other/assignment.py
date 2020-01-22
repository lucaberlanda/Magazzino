import numpy as np
import collections
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df_raw = pd.read_excel('futuro_applications_dataset.xlsx', sheet_name='dataset')
df_raw.columns = df_raw.iloc[0, :]
df_raw = df_raw.drop(0)
# print(df_raw.describe().T)

cmp_list = df_raw.loc[:, 'Company ID'].values.tolist()
cmp_doubles = [item for item, count in collections.Counter(cmp_list).items() if count > 1]
df_raw = df_raw.set_index('Company ID')

for cmp in cmp_doubles:
    df_raw = df_raw.drop(cmp)

df_raw.loc[:, 'Interest Rate Proposed'] = df_raw.loc[:, 'Interest Rate Proposed']. \
    fillna(df_raw.loc[:, 'Interest Rate Proposed'].median())

df_raw.loc[:, 'fdgRating'] = df_raw.loc[:, 'fdgRating'].fillna(df_raw.loc[:, 'fdgRating'].mean())
aggregations = {'applicationDate': 'count', 'Interest Rate Proposed': 'median', 'fdgRating': 'mean'}

# Box Whisker Plot with Interest Rate
# ptf without the rejected / not eligible companies
# the acceptance depend on a variable that you should change
# Plot (AcceptedByTheClient + Financed) vs (Uninterested)
# Random Forest
# The probability

# seems reasonable to assume that the interest rate proposed would be lower than the one originally proposed
y_mapping_dict = {'AcceptedByTheClient': 1,
                  'Financed': 1,
                  'ReadyForApproval': 1,
                  'Uninterested': 0}

do_drop = ['Rejected']
df_approved = df_raw[df_raw.dossierStatus != 'Rejected']
df_tba = df_approved[df_approved.dossierStatus == 'ApprovedByCredimi']
df_answered = df_approved[df_approved.dossierStatus.isin(list(y_mapping_dict.keys()))]
df_answered = df_answered.replace({'dossierStatus': y_mapping_dict})

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
clf = RandomForestClassifier(n_estimators=500, max_depth=5)
clf.fit(X_train, y_train)
clf_predict = clf.predict(X_test)

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
