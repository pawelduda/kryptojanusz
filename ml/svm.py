from sklearn import svm
import numpy

input_btc_eth_file = open('../btc_historical_data/normalized_data/BTC_ETH_normalized.csv')
data = numpy.loadtxt(
    input_btc_eth_file,
    delimiter=',',
    dtype={
        'names': ('first_val', 'second_val', 'diff'),
        'formats': ('<f8', '<f8', 'S15')
    }
)

X = [[row[0], row[1]] for row in data[0:20000]]
y = [row[2] for row in data[0:20000]]
clf = svm.SVC(kernel='poly', verbose=3)
print(clf.fit(X, y))
print('done training')

samples = [
    [0.03919189,0.01031197],
    [0.01031197,0.009],
    [0.009,0.0090],
    [0.00901,0.00686011],
    [0.00686011,0.00560399],
    [0.00560399,0.00557235],
    [0.00557235,0.0070626],
    [0.00706269,0.007992],
    [0.0079928,0.0083207],
    [0.00832073,0.00602666],
    [0.00602666,0.0066016],
    [0.00660165,0.00595616],
    [0.00595616,0.00589533]
]
print(clf.predict(samples))

# current status: stuck, this model is wrong
# pytanie: CO chce przewidziec i funkcja CZEGO jest to cos
