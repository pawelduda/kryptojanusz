from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

from fee import get_fee_amount
import fee

DECIDE_TO_BUY = 1
DECIDE_TO_HOLD = 0
DECIDE_TO_SELL = -1

dataset_file = open('../btc_historical_data/data/BTC_XRP.csv')
dataset = pd.read_csv(dataset_file, sep=',', header=0, low_memory=False)

train_data = [[], []]
for i in range(1, len(dataset)):
    first_price = dataset['close'].values[i - 1]
    next_price = dataset['close'].values[i]

    change = next_price - first_price
    pct_change = change / first_price
    fee_pct = get_fee_amount()
    fee_pct = fee_pct * 2 # Fee x 2 since we'd need to clear both buy and sell fees to be profitable
    fee_pct = fee_pct * fee.FEE_MANAGEMENT_STRATEGY # See desc in settings.py

    if abs(pct_change) < fee_pct:
        decision = DECIDE_TO_HOLD
        decision_str = 'HOLD'
    elif change > 0:
        decision = DECIDE_TO_BUY
        decision_str = 'BUY'
    else:
        decision = DECIDE_TO_SELL
        decision_str = 'SELL'

    train_data[0].append((first_price, next_price))
    train_data[1].append(decision)

    # TODO: create a debug function which takes splat args
    print('*****DEBUG*****')
    print('Row #{}'.format(i))
    print('From {} to {}'.format(first_price, next_price))
    print('Change: {}%'.format(pct_change))
    print(decision_str)
    print('*****\n')

X, y = train_data
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

clf = RandomForestClassifier(
    criterion='entropy',
    max_features=None,
    min_samples_leaf=20,
    n_jobs=-1
)

print(clf.fit(X_train, y_train))
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print('*****DEBUG*****')
print('Score: {}%'.format(score))
print('*****\n')

print('*****DEBUG*****')
cross_val_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
print(cross_val_scores)
print('Accuracy: %0.2f (+/- %0.2f)' % (cross_val_scores.mean(), cross_val_scores.std() * 2))
print('*****\n')

# TODO: somehow plot this
