from sklearn import svm
from sklearn.externals import joblib
import numpy

input_btc_eth_file = open('../btc_historical_data/normalized_data/BTC_ETH_normalized.csv')
data = numpy.loadtxt(
    input_btc_eth_file,
    delimiter=',',
    skiprows=1,
    dtype='float'
)

# X = [row[1:4] for row in data[0:10000]]
# y = [row[13] for row in data[0:10000]]

# clf = svm.SVC(kernel='poly', verbose=3)
# print(clf.fit(X, y))
# print('done training')
# joblib.dump(clf, 'clf.pkl')

clf = joblib.load('clf.pkl')

print(clf.predict([row[1:4] for row in data[4000:5000]]))

# current status: stuck, this model is wrong
# pytanie: CO chce przewidziec i funkcja CZEGO jest to cos
# TODO: sample the dataset
# TODO: visualise the results on plot
