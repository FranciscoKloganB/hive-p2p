import random
import numpy as np


def excluding_randrange(start, exclude_from, exclude_to, stop, step=1):
    if start < exclude_from < exclude_to < stop:
        raise RuntimeError("Ranges aren't properly defined. Convention is start < exclude_from < exclude_to < stop ")

    randint_1 = random.randrange(start, exclude_from, step)  # [start, exclude_from)
    randint_2 = random.randrange(exclude_to, stop, step)  # [exclude_to, stop)

    total_range = stop - start
    range_1 = exclude_from - start
    range_2 = stop - exclude_to

    return np.random.choice(a=[range_1, range_2], p=[range_1/total_range, range_2/total_range])
