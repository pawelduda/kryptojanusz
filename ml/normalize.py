import numpy

# https://www.researchgate.net/publication/222043783_Predicting_direction_of_stock_price_index_movement_using_artificial_neural_networks_and_support_vector_machines_The_sample_of_the_Istanbul_Stock_Exchange
# Features:
# [ ] 10-day moving average
# [ ] weighted 10-day moving average
# [ ] momentum
# [ ] Stochastic K%
# [ ] Stochastic D%
# [ ] Relative Strength Index
# [ ] Moving Average Convergence Divergence
# [ ] Larry William's R%
# [ ] Accumulation/Distribution Oscillator
# [ ] Commodity Channel Index

def change(previousValue, currentValue):
    if previousValue < currentValue:
        return 'decrease'
    elif previousValue == currentValue:
        return 'equal'
    elif previousValue > currentValue:
        return 'increase'

input_btc_eth_file = open('../btc_historical_data/data/BTC_ETH.csv')
input_btc_eth_file.readline()
data = numpy.loadtxt(
    input_btc_eth_file,
    delimiter=',',
    dtype='float'
)

output_file = open('../btc_historical_data/normalized_data/BTC_ETH_normalized.csv', 'w')
print(data[0])
for i, row in enumerate(data[1:]):
    output_file.write(
        # weightedAverage, change (-1 if decrease, 0 if equal, 1 if increase)
        '{0},{1},{2}\n'.format(data[i - 1][-1], data[i][-1], change(data[i - 1][-1], data[i][-1]))
    )

input_btc_eth_file.close()
output_file.close()
