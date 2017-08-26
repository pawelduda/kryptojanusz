import pandas as pd
import talib

DECIDE_TO_BUY = 1
DECIDE_TO_HOLD = 0
DECIDE_TO_SELL = -1

def predict_newest(chart_data):
    dataframe = pd.read_json(chart_data, convert_dates=True)
    dataframe['T'] = pd.to_datetime(dataframe['T'], utc=True)

    # return (int(signal), int(newest_dataset_timestamp))
    close_prices = dataframe['C'].values
    low_prices = dataframe['L'].values
    high_prices = dataframe['H'].values

    sma = talib.SMA(close_prices, timeperiod=20)
    stddev = pd.rolling_std(close_prices, window=20)

    stoch_rsi = talib.STOCHRSI(close_prices, timeperiod=20)
    stoch_rsi_k = stoch_rsi[0]
    stoch_rsi_d = stoch_rsi[1]

    upper_bband = []
    lower_bband = []
    for i in range(len(dataframe)):
        upper_bband.append(sma[i] + (stddev[i] * 2))
        lower_bband.append(sma[i] - (stddev[i] * 2))

    signals = []
    for i in range(len(dataframe)):
        if low_prices[i] <= lower_bband[i] and stoch_rsi_k[i] <= 20.:
            decision = DECIDE_TO_BUY
            # decision_str = 'BUY'
        elif high_prices[i] >= upper_bband[i] and stoch_rsi_k[i] >= 80.:
            decision = DECIDE_TO_SELL
            # decision_str = 'SELL'
        else:
            decision = DECIDE_TO_HOLD
            # decision_str = 'HOLD'

        signals.append(decision)

    newest_signal = signals[-1]
    retrieval_date_gmt = dataframe['T'].values[-1]
    last_price = close_prices[-1]
    last_upper_bband = upper_bband[-1]
    last_lower_bband = lower_bband[-1]
    last_stoch_rsi_k = stoch_rsi_k[-1]

    return (
        int(newest_signal),
        str(retrieval_date_gmt),
        float(last_upper_bband),
        float(last_price),
        float(last_lower_bband),
        float(last_stoch_rsi_k)
    )
