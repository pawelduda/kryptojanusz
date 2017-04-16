from settings import PRINT_DEBUG_FOCUS

def print_debug(*args):
    if not PRINT_DEBUG_FOCUS:
        print('*****DEBUG*****')
        for arg in args:
            print(arg)
        print('***************\n')

# Focused print debug
def fprint_debug(*args):
    print('*****DEBUG*****')
    for arg in args:
        print(arg)
    print('***************\n')
