import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split


ALLdata = pd.read_csv('../kobe-bryant-shot-selection/data.csv')
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
data['loc_x'] = pd.cut(data['loc_x'], 50)
data['loc_y'] = pd.cut(data['loc_y'], 50)
data['lat'] = pd.cut(data['lat'],10)
data['lon'] = pd.cut(data['lon'],10)

#One Hot Coding
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
y_pred1 = clf.fit(X, Y).predict(X_test)
print("Accuracy : %f" % ( (y_test == y_pred1).sum()/X_test.shape[0]))

y_pred = clf.fit(X, Y).predict_proba(data_submit)
submission = pd.DataFrame()
submission["shot_id"] = data_submit.index
submission["shot_made_flag"]= y_pred[:,1]
print(1)