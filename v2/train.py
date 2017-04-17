from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib

from settings import get_fee_amount
from tools import print_debug, fprint_debug
import settings

DECIDE_TO_BUY = 1
DECIDE_TO_HOLD = 0
DECIDE_TO_SELL = -1

dataset = []
scaler = StandardScaler()

def simulate_for_dataset(train_data_pkl_path):
    train_data = joblib.load(train_data_pkl_path)
    X, y = train_data
    # X = scaler.fit_transform(X)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)
    X_train = X
    X_test = X
    y_train = y
    y_test = y

    if settings.CLASSIFIER['train']:
        # clf = RandomForestClassifier(
        #     criterion='entropy',
        #     max_features=None,
        #     min_samples_leaf=20,
        #     n_jobs=-1
        # ) # looks promising
        # Overall the classifiers seem to get better the more data columns they get
        # clf = SVC(kernel='linear', C=0.025)
        # clf = GaussianNB()
        # clf = LinearDiscriminantAnalysis()
        # clf = QuadraticDiscriminantAnalysis()
        clf = AdaBoostClassifier() # looks very promising
        # clf = KNeighborsClassifier(3) # looks shitty
        # clf = KNeighborsClassifier(5) # looks a bit better
        # clf = KNeighborsClassifier(10) # 10 seems to be the most optimal for KNeighbors, however still shitty

        print(clf.fit(X_train, y_train))
        score = clf.score(X_test, y_test)
        print_debug('Score: {}%'.format(score))

        # cross_val_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
        # print_debug(
        #     cross_val_scores,
        #     'Accuracy: %0.2f (+/- %0.2f)' % (cross_val_scores.mean(), cross_val_scores.std() * 2)
        # )
        joblib.dump(clf, 'clf_v2.pkl')

    # ***** Naive simulation *****
    # X_test = scaler.inverse_transform(X_test)

    BTC = 'BTC'
    ALT = 'ALT'

    # Plot data
    prices = []
    actions_taken = []
    projected_btc_balances = []
    sma_values = []
    wma_values = []
    momentum_values = []
    rsi_values = []
    cci_values = []
    stoch_k_values = []
    stoch_d_values = []
    macd_d_values = []
    macd_d_signal_values = []
    macd_d_hist_values = []
    willr_values = []
    adosc_values = []

    transactions_made = 0

    if settings.CLASSIFIER['simulate']:
        clf = joblib.load('clf_v2.pkl')

        # TODO: apply fees
        predicted = clf.predict(X_test)
        balance_btc = 1.
        balance_alt = 0.
        currency_owned = BTC
        previous_projected_btc_balance = None
        projected_btc_balance = balance_btc

        STARTING_INDEX = 0
        AMOUNT_OF_ROWS = 170000
        for i, prediction in enumerate(predicted[STARTING_INDEX:STARTING_INDEX + AMOUNT_OF_ROWS], STARTING_INDEX):
            action_taken = 'n/a'

            previous_alt_price, current_alt_price, sma, wma, momentum, rsi, cci, \
            stoch_k, stoch_d, macd_d, macd_signal, macd_hist, willr, adosc = X_test[i]

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
            actions_taken.append(action_taken)
            sma_values.append(sma)
            wma_values.append(wma)
            momentum_values.append(momentum)
            rsi_values.append(rsi)
            cci_values.append(cci)
            stoch_k_values.append(stoch_k)
            stoch_d_values.append(stoch_d)
            macd_d_values.append(macd_d)
            macd_d_signal_values.append(macd_signal)
            macd_d_hist_values.append(macd_hist)
            willr_values.append(willr)
            adosc_values.append(adosc)

            print_debug(
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

    fprint_debug(
        'Transactions made: {}'.format(transactions_made)
    )

    # ***** PLOT *****
    plot_X = range(0, len(projected_btc_balances))

    figure, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(plot_X, prices)
    axarr[0].set_title('Altcoin price over time (BTC)')

    for i, action_taken in enumerate(actions_taken):
        if action_taken == 'BUY':
            axarr[0].arrow(i, prices[i], 0, 0.005, fc='k', ec='k')
        elif action_taken == 'SELL':
            axarr[0].arrow(i, prices[i], 0, -0.005, fc='k', ec='k')

    axarr[1].plot(plot_X, projected_btc_balances)
    axarr[1].set_title('Simulated BTC balance over time')

    # axarr[2].plot(plot_X, sma_values)
    # axarr[2].set_title('SMA')

    # axarr[3].plot(plot_X, wma_values)
    # axarr[3].set_title('WMA')

    # axarr[4].plot(plot_X, momentum_values)
    # axarr[4].set_title('Momentum')

    # axarr[5].plot(plot_X, rsi_values)
    # axarr[5].set_title('RSI')

    # axarr[6].plot(plot_X, cci_values)
    # axarr[6].set_title('CCI')

    # plt.show()

    plot_filename = train_data_pkl_path.split('.')[0]

    figure.set_size_inches(19.2, 10.8)
    figure.savefig('{}.png'.format(plot_filename), dpi=100)

def simulate_for_all_datasets():
    for source_filename in sorted(glob.glob('btc_historical_data/data/prepared_data/*.pkl')):
        simulate_for_dataset(source_filename)
        fprint_debug('Simulation done for {}'.format(source_filename))

simulate_for_all_datasets()