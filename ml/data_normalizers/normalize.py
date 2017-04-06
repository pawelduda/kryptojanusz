import numpy
import talib
import pandas as pd

from features.simple_10_day_moving_average import simple_10_day_moving_average

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

# close = numpy.random.random(100)
# print(close)
# output = talib.CCI(close)
# print(output)

df = pd.read_csv('../btc_historical_data/data/BTC_ETH.csv', sep=',', header=0, low_memory=False)

print df
sma = talib.SMA(df['close'].values) # TODO: tweak timeperiod to be 10 days
wma = talib.WMA(df['close'].values) # TODO: tweak timeperiod to be 10 days
mom = talib.MOM(df['close'].values)
stoch_k, stoch_d = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
rsi = talib.RSI(df['close'].values)
macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
willr = talib.WILLR(df['high'].values, df['low'].values, df['close'].values)
adosc = talib.ADOSC(df['high'].values, df['low'].values, df['close'].values, df['quoteVolume'].values)
cci = talib.CCI(df['high'].values, df['low'].values, df['close'].values)


# input_btc_eth_file = open('../btc_historical_data/data/BTC_ETH.csv')
# input_btc_eth_file.readline()
# data = numpy.loadtxt(
#     input_btc_eth_file,
#     delimiter=',',
#     dtype={
#         'names': ('date', 'high', 'low', 'open', 'close', 'volume', 'quoteVolume', 'weightedAverage'),
#         'formats': ('int', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8', '<f8')
#     }
# )

output_file = open('../btc_historical_data/normalized_data/BTC_ETH_normalized.csv', 'w')
for i in range(0, len(df)):
    output_file.write(
        '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n'.format(
            # Features
            sma[i],
            wma[i],
            mom[i],
            stoch_k[i],
            stoch_d[i],
            rsi[i],
            macd[i],
            macd_signal[i],
            macd_hist[i],
            willr[i],
            adosc[i],
            cci[i],
            # Labels
            df['close'].values[i] if i + 10 >= len(df) else df['close'].values[i + 10]
        )
    )

#     last_ten_day_close_values = []


# ten_days_seconds = 864000
# first_entry_timestamp = data[0][0]

# for i, row in enumerate(data[1:]):
#     last_ten_day_close_values = []

#     print(i)

#     # get all values for last 10 days
#     # TODO: this should be optimized
#     current_timestamp = row[0]
#     for j in range(i, 0, -1):
#         if data[i][0] - data[j][0] <= 864000:
#             last_ten_day_close_values.append(data[j][4])
#         else:
#             break

#     if len(last_ten_day_close_values) > 0:
#         ten_day_moving_average = simple_10_day_moving_average(last_ten_day_close_values)
#     else:
#         ten_day_moving_average = 'n/a'

#     output_file.write(
#         # weightedAverage, change (-1 if decrease, 0 if equal, 1 if increase)
#         '{0},{1},{2},{3}\n'.format(
#             data[i - 1][-1],
#             data[i][-1],
#             ten_day_moving_average,
#             data[i][7] # weightedAverage
#         )
#     )

# input_btc_eth_file.close()
# output_file.close()
