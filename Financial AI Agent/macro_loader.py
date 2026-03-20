import random


def get_macro_data():

    inflation = round(random.uniform(2, 7), 2)
    interest = round(random.uniform(3, 8), 2)

    return inflation, interest
