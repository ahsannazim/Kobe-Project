import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np

ALLdata = pd.read_csv('/Users/caiyeming/Downloads/kobe-bryant-shot-selection/data.csv')
missing_value = ALLdata['shot_made_flag'].isnull()

data = ALLdata.copy() # create a copy of data frame
ALL_Y = data['shot_made_flag'].copy()

data.drop('team_name', axis=1, inplace=True) # Always LA Lakers
data.drop('team_id', axis=1, inplace=True) # Always one number
data.drop('shot_made_flag', axis=1, inplace=True)


# Remaining time
data['seconds_from_period_end'] = 60 * data['minutes_remaining'] + data['seconds_remaining']

data.drop('minutes_remaining', axis=1, inplace=True)
data.drop('seconds_remaining', axis=1, inplace=True)

## Matchup - (away/home)
data['away/home'] = data['matchup'].str.contains('vs').astype('int')
data.drop('matchup', axis=1, inplace=True)

# Game date
data['game_date'] = pd.to_datetime(data['game_date'])
data['game_year'] = data['game_date'].dt.year
data['game_month'] = data['game_date'].dt.month
data['game_day'] = data['game_date'].dt.day
data.drop('game_date', axis=1, inplace=True)

# Partition interval
data['loc_x'] = pd.cut(data['loc_x'], 5)#[-250，250] is divided into 5 sections
data['loc_y'] = pd.cut(data['loc_y'], 5)
data['lat'] = pd.cut(data['lat'],10)
data['lon'] = pd.cut(data['lon'],10)
data['shot_distance'] = pd.cut(data['shot_distance'], 5)
data['seconds_from_period_end'] = pd.cut(data['seconds_from_period_end'], 5)


# data_remov0 = data.copy()
# data_remov1 = data.drop('game_day',axis = 1, inplace = False)
# data_remov2 = data_remov1.drop('game_id',axis = 1, inplace = False)
# data_remov3 = data_remov2.drop('shot_zone_range',axis = 1, inplace = False)
# data_remov4 = data_remov3.drop('shot_type',axis = 1, inplace = False)

#One Hot Coding
# categorial_cols = list(data.columns.values)
# for col in categorial_cols:
#     data_remov0 = data.drop(col,axis = 1, inplace = False)
#     categorial_col = list(data_remov0.columns.values)
#     for cols in categorial_col:
#         dummies = pd.get_dummies(data_remov0[cols])
#         dummies = dummies.add_prefix("{}_".format(cols))
#         data_remov0.drop(cols, axis=1, inplace=True)
#         data_remov0 = data_remov0.join(dummies)
#
#     X = data_remov0[~missing_value]
#     Y = ALL_Y[~missing_value]
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.75, random_state=123)
#     clf = BernoulliNB()
#     y_pred1 = clf.fit(X_train, y_train).predict(X_test)
#     print(col, "  Accuracy : %f" % ((y_test == y_pred1).sum() / X_test.shape[0]))

#data.drop('shot_type', axis=1, inplace=True)
#data.drop('period', axis=1, inplace=True)
# data.drop('game_month', axis=1, inplace=True)
# data.drop('combined_shot_type', axis=1, inplace=True)
# data.drop('season', axis=1, inplace=True)
# data.drop('game_year', axis=1, inplace=True)
# data.drop('opponent', axis=1, inplace=True)
# data.drop('away/home', axis=1, inplace=True)
# data.drop('lat', axis=1, inplace=True)
# data.drop('loc_y', axis=1, inplace=True)
data.drop('game_day', axis=1, inplace=True)
data.drop('game_event_id', axis=1, inplace=True)
data.drop('shot_zone_area', axis=1, inplace=True)
data.drop('lon', axis=1, inplace=True)
data.drop('loc_x', axis=1, inplace=True)
data.drop('shot_zone_basic', axis=1, inplace=True)
data.drop('shot_zone_range', axis=1, inplace=True)
data.drop('game_id', axis=1, inplace=True)

categorial_cols = list(data.columns.values)
for col in categorial_cols:
    dummies = pd.get_dummies(data[col])
    dummies = dummies.add_prefix("{}_".format(col))
    data.drop(col, axis=1, inplace=True)
    data = data.join(dummies)


data_submit = data[missing_value]


# Separate dataset for training
X = data[~missing_value]
Y = ALL_Y[~missing_value]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.75, random_state=123)
clf = BernoulliNB()
y_pred1 = clf.fit(X_train, y_train).predict(X_test)
print("Accuracy : %f" % ( (y_test == y_pred1).sum()/X_test.shape[0]))

LR = LogisticRegression()
y_pred2 = LR.fit(X_train, y_train).predict(X_test)
print("Accuracy LogisticRegression: %f" % ((y_test == y_pred2).sum() / X_test.shape[0]))

model = RandomForestRegressor()
model.fit(X, Y)

# feature_rfr = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["importance"])
# feature_rfr_20 = feature_rfr.sort_values("importance", ascending=False).head(20).index
# feature_rfr_20
#
# big_feature = []
# thresholds = [0.90, 0.80, 0.70,0.60,0.50]
# for threshold in thresholds:
#     vt = VarianceThreshold().fit(X)
#     # Find feature names
#     feature_var = data.columns[vt.variances_ > threshold * (1-threshold)]
#
#     features = np.hstack([
#         feature_var,
#         feature_rfr_20,
#     ])
#     features = np.unique(features)
#     big_feature.append(features)
#     data_copy = data.ix[:, features]
#     data_submit_copy = data_submit.ix[:, features]
#     X_copy = X.ix[:, features]
#
#     X_train, X_test, y_train, y_test = train_test_split(X_copy, Y, test_size=0.75, random_state=123)
#
#     clf = BernoulliNB()
#     y_pred1 = clf.fit(X_copy, Y).predict(X_test)
#     print("Accuracy BernoulliNB: %f" % ((y_test == y_pred1).sum() / X_test.shape[0]))
#
#     LR = LogisticRegression()
#     y_pred2 = LR.fit(X_copy, Y).predict(X_test)
#     print("Accuracy LogisticRegression: %f" % ((y_test == y_pred2).sum() / X_test.shape[0]))




# rfe = RFE(LogisticRegression(), 20)
# rfe.fit(X, Y)
#
# rfecv = RFECV(estimator=LogisticRegression(), cv=KFold(n_splits=3, random_state=1), scoring='accuracy', min_features_to_select= 20)
# rfecv.fit(X, Y)
#
# feature_rfe_scoring = pd.DataFrame({
#         'feature': X.columns,
#         'score': rfecv.ranking_
#     })
#
# feat_rfe_20 = feature_rfe_scoring[feature_rfe_scoring['score'] == 1]['feature'].values
# feat_rfe_20





X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.75, random_state=123)
y_pred1 = model.predict(X_test)
print("Accuracy : %f" % ( (y_test == y_pred1).sum()/X_test.shape[0]))



y_pred = clf.fit(X, Y).predict_proba(data_submit)
submission = pd.DataFrame()
submission["shot_id"] = data_submit.index
submission["shot_made_flag"]= y_pred[:,1]
print(1)