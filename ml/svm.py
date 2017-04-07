from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import numpy

input_btc_eth_file = open('../btc_historical_data/normalized_data/BTC_ETH_normalized.csv')
data = numpy.loadtxt(
    input_btc_eth_file,
    delimiter=',',
    skiprows=1,
    dtype='float'
)

features = [row[1:4] for row in data]
targets = [row[13] for row in data]

X_train, X_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.4, random_state=0
)

# X = [row[1:4] for row in data[0:10000]]
# y = [row[13] for row in data[0:10000]]

# clf = svm.SVC(kernel='poly', verbose=3)
# print(clf.fit(X_train, y_train))
# print('done training')
# joblib.dump(clf, 'clf.pkl')

clf = joblib.load('clf.pkl')

# print(clf.predict([row[1:4] for row in data[4000:5000]]))
print(clf.score(X_test, y_test))

# TODO: sample the dataset
# TODO: visualise the results on plot
