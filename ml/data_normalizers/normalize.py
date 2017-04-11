import talib
import pandas as pd

from features.simple_10_day_moving_average import simple_10_day_moving_average

def future_change(current_price, future_price):
    if current_price > future_price:
        return -1
    elif current_price == future_price:
        return 0
    elif current_price < future_price:
        return 1

def normalize(input_dataset_filename, output_dataset_filename):
    # TODO: ensure that weightedAverage is calculated based on last 10 days
    # https://www.researchgate.net/publication/222043783_Predicting_direction_of_stock_price_index_movement_using_artificial_neural_networks_and_support_vector_machines_The_sample_of_the_Istanbul_Stock_Exchange
    # http://ta-lib.org/function.html
    # https://cryptotrader.org/talib
    # Features normalization status:
    # [/] 10-day moving average
    # [/] weighted 10-day moving average
    # [/] momentum
    # [/] Stochastic K%
    # [/] Stochastic D%
    # [/] Relative Strength Index
    # [/] Moving Average Convergence Divergence
    # [/] Larry William's R%
    # [/] Accumulation/Distribution Oscillator
    # [/] Commodity Channel Index
    #
    # Legend:
    # / - I put in numbers and different numbers come out
    # X - done

    # sma = talib.SMA(df['close'].values) # TODO: tweak timeperiod to be 10 days
    # wma = talib.WMA(df['close'].values) # TODO: tweak timeperiod to be 10 days
    # mom = talib.MOM(df['close'].values)
    # stoch_k, stoch_d = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
    # rsi = talib.RSI(df['close'].values)
    # macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
    # willr = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)
    # adosc = talib.ADOSC(df['high'].values, df['low'].values, df['close'].values, df['quote_volume'].values)
    # cci = talib.CCI(df['high'].values, df['low'].values, df['close'].values)

    df = pd.read_csv(input_dataset_filename, sep=',', header=0, low_memory=False)

    output_file = open(output_dataset_filename, 'w')
    for i in range(2880, len(df)):
        if i + 280 >= len(df):
            break

        print(i)

        # 2880 - amount of 5-minute periods in 10 days
        close = df['close'].values[i - 2880:i]
        low = df['low'].values[i - 2880:i]
        high = df['high'].values[i - 2880:i]
        quote_volume = df['quoteVolume'].values[i - 2880:i]

        sma = talib.SMA(close)[-1]
        wma = talib.WMA(close)[-1]
        mom = talib.MOM(close)[-1]

        stoch = talib.STOCH(high, low, close)
        stoch_k = stoch[0][-1]
        stoch_d = stoch[1][-1]

        rsi = talib.RSI(close)[-1]

        macd = talib.MACD(close)
        macd_d = macd[0][-1]
        macd_signal = macd[1][-1]
        macd_hist = macd[2][-1]

        willr = talib.WILLR(high, low, close)[-1]
        adosc = talib.ADOSC(high, low, close, quote_volume)[-1]
        cci = talib.CCI(high, low, close)[-1]

        output_file.write(
            '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17}\n'.format(
                # Features
                sma,
                wma,
                mom,
                stoch_k,
                stoch_d,
                rsi,
                macd_d,
                macd_signal,
                macd_hist,
                willr,
                adosc,
                cci,
                # Misc
                df['close'].values[i + 12],
                # Classes
                future_change(df['close'].values[i], df['close'].values[i + 1]),
                future_change(df['close'].values[i], df['close'].values[i + 12]),
                future_change(df['close'].values[i], df['close'].values[i + 280]),
                # Misc
                df['close'].values[i],
                df['date'].values[i]
            )
        )

    output_file.close()

input_output_dataset_paths = [
    ('../btc_historical_data/data/BTC_ETH.csv', '../btc_historical_data/normalized_data/BTC_ETH_normalized.csv'),
    ('../btc_historical_data/data/BTC_DASH.csv', '../btc_historical_data/normalized_data/BTC_DASH_normalized.csv'),
    ('../btc_historical_data/data/BTC_LTC.csv', '../btc_historical_data/normalized_data/BTC_LTC_normalized.csv')
]

for input_dataset_filename, output_dataset_filename in input_output_dataset_paths:
    normalize(input_dataset_filename, output_dataset_filename)
