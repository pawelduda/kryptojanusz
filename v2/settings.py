# Debug settings
PRINT_DEBUG_FOCUS = False

# Classifier settings
CLASSIFIER = {
    'prepare_dataset': True,
    'train': False,
    'simulate': True
}

# https://poloniex.com/fees/
FEES = {
    'poloniex': [
        {'maker': 0.0015, 'taker': 0.0025, 'volume': '< 600 BTC'},
        {'maker': 0.0014, 'taker': 0.0024, 'volume': '>= 600 BTC'},
        {'maker': 0.0012, 'taker': 0.0022, 'volume': '>= 1200 BTC'},
        {'maker': 0.001, 'taker': 0.0020, 'volume': '>= 2400 BTC'},
        {'maker': 0.0008, 'taker': 0.0016, 'volume': '>= 6000 BTC'},
        {'maker': 0.0005, 'taker': 0.0014, 'volume': '>= 12000 BTC'},
        {'maker': 0.0002, 'taker': 0.0012, 'volume': '>= 18000 BTC'},
        {'maker': 0.0000, 'taker': 0.0010, 'volume': '>= 24000 BTC'},
        {'maker': 0.0000, 'taker': 0.0008, 'volume': '>= 60000 BTC'},
        {'maker': 0.0000, 'taker': 0.0005, 'volume': '>= 120000 BTC'},
    ]
}

TRADE_VOLUME_TRAILING_30_DAYS = '< 600 BTC' # TODO - in the future, make this calculated dynamically
TRADE_MODE = 'taker' # TODO -- in the future, make this more dynamicly chosen via trade.py

# sets how aggressive user wants to be in managing their trades
# 1 = trade when if & when projected profit >= fees
# 2 = trade when if & when projected profit >= 2x fees
# . . .
# n = trade when if & when projected profit >= nx fees
FEE_MANAGEMENT_STRATEGY = 1

def get_fee_amount(volume=TRADE_VOLUME_TRAILING_30_DAYS, mode=TRADE_MODE):
    for meta in FEES['poloniex']:
        if meta['volume'] == volume:
            return meta[mode]

    raise 'could not find fee amount for {} / {}'.format(volume, mode)
