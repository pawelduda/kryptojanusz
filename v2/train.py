from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

from settings import get_fee_amount
from tools import print_debug, fprint_debug
import settings

DECIDE_TO_BUY = 1
DECIDE_TO_HOLD = 0
DECIDE_TO_SELL = -1

dataset = []
if settings.CLASSIFIER['prepare_dataset']:
    dataset_file = open('../btc_historical_data/data/BTC_ETH.csv')
    dataset = pd.read_csv(dataset_file, sep=',', header=0, low_memory=False)

    train_data = [[], []]
    for i in range(1, len(dataset)):
        first_price = dataset['close'].values[i - 1]
        next_price = dataset['close'].values[i]

        change = next_price - first_price
        pct_change = change / first_price
        fee_pct = get_fee_amount()
        fee_pct = fee_pct * 2 # Fee x 2 since we'd need to clear both buy and sell fees to be profitable
        fee_pct = fee_pct * settings.FEE_MANAGEMENT_STRATEGY # See desc in settings.py

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

        print_debug(
            'Row #{}'.format(i),
            'From {} to {}'.format(first_price, next_price),
            'Change: {}%'.format(pct_change),
            decision_str
        )

    X, y = train_data
    # X = StandardScaler().fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)
    X_train = X
    X_test = X
    y_train = y
    y_test = y

if settings.CLASSIFIER['train']:
    clf = RandomForestClassifier(
        criterion='entropy',
        max_features=None,
        min_samples_leaf=20,
        n_jobs=-1
    )

    print(clf.fit(X_train, y_train))
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print_debug('Score: {}%'.format(score))

    cross_val_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    print_debug(
        cross_val_scores,
        'Accuracy: %0.2f (+/- %0.2f)' % (cross_val_scores.mean(), cross_val_scores.std() * 2)
    )

    joblib.dump(clf, 'clf.pkl')

# ***** Naive simulation *****

BTC = 'BTC'
ALT = 'ALT'
prices = []
projected_btc_balances = []
transactions_made = 0

if settings.CLASSIFIER['simulate']:
    clf = joblib.load('clf.pkl')

    # TODO: apply fees
    predicted = clf.predict(X_test)
    balance_btc = 1.
    balance_alt = 0.
    currency_owned = BTC
    previous_projected_btc_balance = None
    projected_btc_balance = balance_btc

    for i, prediction in enumerate(predicted[90000:92000], 90000):
        action_taken = 'n/a'
        previous_alt_price, current_alt_price = X_test[i]

        previous_projected_btc_balance = projected_btc_balance

        if prediction == DECIDE_TO_HOLD:
            action_taken = 'HOLD'

            if currency_owned == BTC:
                balance_btc # does not change
                projected_btc_balance = balance_btc
            elif currency_owned == ALT:
                balance_alt = (balance_alt / previous_alt_price) * current_alt_price
                projected_btc_balance = balance_alt * current_alt_price

        elif prediction == DECIDE_TO_BUY and currency_owned == BTC:
            action_taken = 'BUY'

            balance_alt = balance_btc / current_alt_price
            balance_btc = 0.
            currency_owned = ALT

            projected_btc_balance = balance_alt * current_alt_price

            transactions_made += 1
        elif prediction == DECIDE_TO_SELL and currency_owned == ALT:
            action_taken = 'SELL'

            balance_btc = balance_alt * current_alt_price
            balance_alt = 0.
            currency_owned = BTC

            projected_btc_balance = balance_btc

            transactions_made += 1

        if action_taken == 'n/a':
            if currency_owned == BTC:
                projected_btc_balance = balance_btc
            elif currency_owned == ALT:
                projected_btc_balance = balance_alt * current_alt_price

        # Gather data for plot
        projected_btc_balances.append(projected_btc_balance)
        prices.append(current_alt_price)

        fprint_debug(
            'Alt price change: {}'.format(X_test[i]),
            'Action taken: {}'.format(action_taken),
            'BTC balance after decision: {}'.format(balance_btc),
            'Alt balance after decision: {}'.format(balance_alt),
            'Previous projected BTC balance: {}'.format(previous_projected_btc_balance),
            'Projected BTC balance: {}'.format(projected_btc_balance),
            'Currency owned: {}'.format(currency_owned),
            'Rows analyzed: {}'.format(i),
            'Transactions made: {}'.format(transactions_made)
        )

        if balance_btc == 0. and balance_alt == 0.:
            raise 'nope'

# ***** PLOT *****
plot_X = range(0, len(projected_btc_balances))

f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(plot_X, prices)
axarr[0].set_title('Altcoin price over time (BTC)')

axarr[1].plot(plot_X, projected_btc_balances)
axarr[1].set_title('Simulated BTC balance over time')

plt.show()
