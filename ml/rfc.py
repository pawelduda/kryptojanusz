import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import numpy

input_btc_eth_file = open('../btc_historical_data/normalized_data/BTC_ETH_normalized.csv')
data = numpy.loadtxt(
    input_btc_eth_file,
    delimiter=',',
    skiprows=1,
    dtype='float'
)

features = [row[1:12] for row in data[0:173000]]
targets = [row[13] for row in data[0:173000]]

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=0
)


clf = RandomForestClassifier(
    criterion='entropy',
    max_features=None,
    min_samples_leaf=20,
    n_jobs=-1
)
print(clf.fit(X_train, y_train))
print('done training')
joblib.dump(clf, 'clf.pkl')

clf = joblib.load('clf.pkl')

cross_val_scores = cross_val_score(clf, features, targets, cv=5)
print(cross_val_scores)
print('Accuracy: %0.2f (+/- %0.2f)' % (cross_val_scores.mean(), cross_val_scores.std() * 2))
# predicted = clf.predict(features)
# predicted = cross_val_predict(clf, features, targets, cv=5)
# print(metrics.accuracy_score(targets, predicted))

print(clf.score(X_test, y_test))

# TODO: sample the dataset
# TODO: visualise the results on plot

# http://stackoverflow.com/questions/31681373/making-svm-run-faster-in-python
# Best:
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
#             max_depth=None, max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=20,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=10, n_jobs=-1, oob_score=False, random_state=None,
#             verbose=0, warm_start=False)
# done training
# 0.698901734104

# Prediction accuracy:
#
# in 5 mins:
# [ 0.29003772  0.44358022  0.48284937  0.46777401  0.48166452]
# Accuracy: 0.43 (+/- 0.15) <-- cross validation score
# 0.477951927013 <-- single test split score (1/5 of dataset)
#
# in 1 hour:
# [ 0.45879876  0.49818703  0.50590678  0.50270492  0.50412329]
# Accuracy: 0.49 (+/- 0.04) <-- cross validation score
# 0.565559389438 <-- single test split score (1/5 of dataset)
#
# in 1 day:
# [ 0.42672086  0.46686941  0.48472088  0.47662076  0.48475012]
# Accuracy: 0.47 (+/- 0.04) <-- cross validation score
# 0.74241183695 <-- single test split score (1/5 of dataset)
