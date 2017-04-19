from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import glob, os
import pandas as pd
import sys
import talib

from settings import get_fee_amount
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
    LOOKBACK_DELTA = 288
    train_data = [[], []]

    for i in range(LOOKBACK_DELTA, len(dataset)):
        recent_close_prices = dataset['close'].values[i - LOOKBACK_DELTA:i]
        recent_high_prices = dataset['high'].values[i - LOOKBACK_DELTA:i]
        recent_low_prices = dataset['low'].values[i - LOOKBACK_DELTA:i]
        recent_quote_volume = dataset['quoteVolume'].values[i - LOOKBACK_DELTA:i]

        first_price = dataset['close'].values[i - 1]
        next_price = dataset['close'].values[i]

        sma = talib.SMA(recent_close_prices)[-1]
        wma = talib.WMA(recent_close_prices)[-1]
        momentum = talib.MOM(recent_close_prices)[-1]
        rsi = talib.RSI(recent_close_prices)[-1]
        cci = talib.CCI(recent_high_prices, recent_low_prices, recent_close_prices)[-1]

        stoch = talib.STOCH(recent_high_prices, recent_low_prices, recent_close_prices)
        stoch_k = stoch[0][-1]
        stoch_d = stoch[1][-1]

        macd = talib.MACD(recent_close_prices)
        macd_d = macd[0][-1]
        macd_signal = macd[1][-1]
        macd_hist = macd[2][-1]

        willr = talib.WILLR(recent_high_prices, recent_low_prices, recent_close_prices)[-1]
        adosc = talib.ADOSC(recent_high_prices, recent_low_prices, recent_close_prices, recent_quote_volume)[-1]

        obv = talib.OBV(recent_close_prices, recent_quote_volume)[-1]
        aroon = talib.AROON(recent_high_prices, recent_low_prices)
        aroon_down = aroon[0][-1]
        aroon_up = aroon[1][-1]
        atr = talib.ATR(recent_high_prices, recent_low_prices, recent_close_prices)[-1]

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

        train_data[0].append(
            (
                first_price, next_price, sma, wma, momentum, rsi, cci, stoch_k, stoch_d,
                macd_d, macd_signal, macd_hist, willr, adosc, obv, aroon_down, aroon_up, atr
            )
        )
        train_data[1].append(decision)

        print_debug(
            'Row #{}'.format(i),
            'From {} to {}'.format(first_price, next_price),
            'Change: {}%'.format(pct_change),
            decision_str
        )

    joblib.dump(train_data, output_dataset_path)

def prepare_all_datasets():
    os.chdir('btc_historical_data/data')

    if sys.argv[1] == '1':
        for source_filename in sorted(glob.glob('*.csv'))[0:40]:
            target_filename = source_filename.split('.')[0]
            prepare_dataset(source_filename, 'prepared_data/{}.pkl'.format(target_filename))
            fprint_debug('Prepared dataset for {}'.format(target_filename))
    elif sys.argv[1] == '2':
        for source_filename in sorted(glob.glob('*.csv'))[41:-1]:
            target_filename = source_filename.split('.')[0]
            prepare_dataset(source_filename, 'prepared_data/{}.pkl'.format(target_filename))
            fprint_debug('Prepared dataset for {}'.format(target_filename))

prepare_all_datasets()
# prepare_dataset('btc_historical_data/data/BTC_ETH.csv', 'btc_historical_data/data/prepared_data/BTC_ETH.pkl')
