from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import glob, os
import pandas as pd
import sys
import talib

from settings import get_fee_amount_poloniex, get_fee_amount_bittrex
from tools import print_debug, fprint_debug
import settings

DECIDE_TO_BUY = 1
DECIDE_TO_HOLD = 0
DECIDE_TO_SELL = -1

scaler = StandardScaler()

def prepare_dataset(dataset_path, output_dataset_path):
    dataset_file = open(dataset_path)
    dataset = pd.read_csv(dataset_file, sep=',', header=0, low_memory=False)

    # 288 = 1 day with historical data, will have to be tweaked for the real data
    # LOOKBACK_DELTA = 288 * 20 # 20 days required for BBANDS
    train_data = [[], []]

    close_prices = dataset['close'].values
    high_prices = dataset['high'].values
    low_prices = dataset['low'].values
    quote_volume = dataset['quoteVolume'].values

    # talib.BBANDS appears to be broken: https://github.com/mrjbq7/ta-lib/issues/151, need to calculate manually
    # bbands = talib.BBANDS(close_prices, nbdevup=2.0, nbdevdn=2.0)
    sma = talib.SMA(close_prices, timeperiod=20)
    stddev = pd.rolling_std(close_prices, window=20)

    stoch_rsi = talib.STOCHRSI(close_prices, timeperiod=20)
    stoch_rsi_k = stoch_rsi[0]
    stoch_rsi_d = stoch_rsi[1]

    upper_bband = []
    lower_bband = []
    for i in range(len(dataset)):
        upper_bband.append(sma[i] + (stddev[i] * 2))
        lower_bband.append(sma[i] - (stddev[i] * 2))

    decisions = []
    for i in range(len(dataset)):
        if low_prices[i] <= lower_bband[i] and stoch_rsi_k[i] <= 20.:
            decision = DECIDE_TO_BUY
            # decision_str = 'BUY'
        elif high_prices[i] >= upper_bband[i] and stoch_rsi_k[i] >= 80.:
            decision = DECIDE_TO_SELL
            # decision_str = 'SELL'
        else:
            decision = DECIDE_TO_HOLD
            # decision_str = 'HOLD'

        decisions.append(decision)

    for i in range(20, len(dataset)):
        train_data[0].append(
            (
                high_prices[i], low_prices[i], close_prices[i], upper_bband[i], lower_bband[i], stoch_rsi_k[i]
            )
        )
        train_data[1].append(decisions[i])

    # print_debug(
        # 'Row {}'.format(i),
        # 'Current price: {}'.format(current_price),
        # 'Stoch RSI: {}'.format(stoch_rsi_k),
        # 'Upper BBAND: {}'.format(upper_band),
        # 'Lower BBAND: {}'.format(lower_band),
        # decision_str
    # )

    joblib.dump(train_data, output_dataset_path)

def prepare_all_datasets():
    os.chdir('btc_historical_data/data')

    for source_filename in sorted(glob.glob('*.csv')):
        target_filename = source_filename.split('.')[0]
        prepare_dataset(source_filename, 'prepared_data/{}.pkl'.format(target_filename))
        fprint_debug('Prepared dataset for {}'.format(target_filename))

    # if sys.argv[1] == '1':
        # for source_filename in sorted(glob.glob('*.csv'))[0:40]:
            # target_filename = source_filename.split('.')[0]
            # prepare_dataset(source_filename, 'prepared_data/{}.pkl'.format(target_filename))
            # fprint_debug('Prepared dataset for {}'.format(target_filename))
    # elif sys.argv[1] == '2':
        # for source_filename in sorted(glob.glob('*.csv'))[41:-1]:
            # target_filename = source_filename.split('.')[0]
            # prepare_dataset(source_filename, 'prepared_data/{}.pkl'.format(target_filename))
            # fprint_debug('Prepared dataset for {}'.format(target_filename))

prepare_all_datasets()
# prepare_dataset('btc_historical_data/data/BTC_ETH.csv', 'btc_historical_data/data/prepared_data/BTC_ETH.pkl')
