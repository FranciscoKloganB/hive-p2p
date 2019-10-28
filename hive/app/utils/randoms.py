import random
import numpy as np
import logging

def excluding_randrange(start, exclude_from, exclude_to, stop, step=1):
    if (exclude_from <= start) or (stop <= exclude_to) or (exclude_to <= exclude_from):
        logging.error("{} < {} < {} < {}\n".format(start, exclude_from, exclude_to, stop))
        raise ValueError("Ranges aren't properly defined. Convention is start < exclude_from < exclude_to < stop ")

    randint_1 = random.randrange(start, exclude_from, step)  # [start, exclude_from)
    randint_2 = random.randrange(exclude_to, stop, step)  # [exclude_to, stop)

    range_1_size = exclude_from - start
    range_2_size = stop - exclude_to
    unified_size = range_1_size + range_2_size

    return np.random.choice(a=[randint_1, randint_2], p=[range_1_size/unified_size, range_2_size/unified_size]).item()
