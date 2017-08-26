from io import StringIO
from sklearn.externals import joblib
import numpy as np
import os
import pandas as pd
import talib
import time

from settings import get_fee_amount
from tools import print_debug, fprint_debug
import settings

LOOKBACK_DELTA = 288

# first_price, next_price, sma, wma, momentum, rsi, cci, stoch_k, stoch_d,
# macd_d, macd_signal, macd_hist, willr, adosc, obv, aroon_down, aroon_up, atr
def predict_newest(poloniex_api_url, columns, currency_pair):
    dataset = pd.read_json(poloniex_api_url, convert_dates=False)
    dataset = dataset.to_csv(index=False, columns=columns)
    dataset = StringIO(unicode(dataset))
    dataset = pd.read_csv(dataset, sep=',', header=0, low_memory=False)

    # Must be absolute path
    os.chdir('/home/pawelduda/kryptojanusz/')
    clf = joblib.load('clf_v2.pkl')

    features = []
    # newest_dataset_timestamp = dataset['date'].values[-1]
    newest_dataset_timestamp = time.time()

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

        # change = next_price - first_price
        # pct_change = change / first_price
        # fee_pct = get_fee_amount()
        # fee_pct = fee_pct * 2 # Fee x 2 since we'd need to clear both buy and sell fees to be profitable
        # fee_pct = fee_pct * settings.FEE_MANAGEMENT_STRATEGY # See desc in settings.py

        features.append(
            (
                first_price, next_price, sma, wma, momentum, rsi, cci, stoch_k, stoch_d,
                macd_d, macd_signal, macd_hist, willr, adosc, obv, aroon_down, aroon_up, atr
            )
        )


    # ##### MOST RECENT DATA SAMPLE PART START #####
    # ticker_columns
    # [baseVolume, high24hr, highestBid, id, isFrozen, last, low24hr, lowestAsk, percentChange, quoteVolume]
    ticker_data = pd.read_json('https://poloniex.com/public?command=returnTicker')

    recent_close_prices = np.append(
        dataset['close'].values[-LOOKBACK_DELTA:],
        ticker_data[currency_pair]['last']
    )
    recent_high_prices = np.append(
        dataset['high'].values[-LOOKBACK_DELTA:],
        # TODO: seems kinda weak, let's see how it goes
        ticker_data[currency_pair]['highestBid'] # this seemed to have caused too much chaos, rapidly losing money
    )
    recent_low_prices = np.append(
        dataset['low'].values[-LOOKBACK_DELTA:],
        # TODO: seems kinda weak, let's see how it goes
        ticker_data[currency_pair]['lowestAsk'] # this seemed to have caused too much chaos, rapidly losing money
    )
    recent_quote_volume = np.append(
        dataset['quoteVolume'].values[-LOOKBACK_DELTA:],
        # TODO: right now I have no idea how to get recent quoteVolume
        recent_quote_volume[-1]
    )

    first_price = features[-1][1]
    next_price = ticker_data[currency_pair]['last']

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

    features.append(
        (
            first_price, next_price, sma, wma, momentum, rsi, cci, stoch_k, stoch_d,
            macd_d, macd_signal, macd_hist, willr, adosc, obv, aroon_down, aroon_up, atr
        )
    )
    # ##### MOST RECENT DATA SAMPLE PART END #####

    # TODO: verify whether we are gathering data with proper indexes (ranges) BOTH here and in train.py!!!
    # print(features[-1])
    prediction = clf.predict([features[-1]])
    return (int(prediction), int(newest_dataset_timestamp))