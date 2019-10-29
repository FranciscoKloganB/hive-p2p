import random
import numpy as np
import logging

def excluding_randrange(start, stop, start_again, stop_again, step=1):
    if (stop <= start) or (stop_again <= start_again) or (start_again <= stop):
        logging.error("{} < {} < {} < {}\n".format(start, stop, start_again, stop_again))
        raise ValueError("Ranges aren't properly defined. Convention is start < exclude_from < exclude_to < stop ")

    randint_1 = random.randrange(start, stop, step)  # [start, exclude_from)
    randint_2 = random.randrange(start_again, stop_again, step)  # [exclude_to, stop)

    range_1_size = stop - start
    range_2_size = stop_again - start_again
    unified_size = range_1_size + range_2_size

    return np.random.choice(a=[randint_1, randint_2], p=[range_1_size/unified_size, range_2_size/unified_size]).item()
