import sklearn.metrics as metrics
from sklearn import svm
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

features = [row[4:6] for row in data[0:10000]]
targets = [row[13] for row in data[0:10000]]

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.4, random_state=0
)


clf = svm.SVC(kernel='poly', verbose=3)
print(clf.fit(X_train, y_train))
print('done training')
joblib.dump(clf, 'clf.pkl')

clf = joblib.load('clf.pkl')

# cross_val_score(clf, features, targets, cv=5))
predicted = clf.predict([row[4:6] for row in data[0:150000]])
# predicted = cross_val_predict(clf, [row[4:6] for row in data[0:100]], [row[13] for row in data[0:100]], cv=5)
print(metrics.accuracy_score([row[13] for row in data[0:150000]], predicted))


# print(clf.score(X_test, y_test))

# TODO: sample the dataset
# TODO: visualise the results on plot
