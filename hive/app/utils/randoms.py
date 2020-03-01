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


def random_index(i: int, size: int) -> int:
    """
    Returns a random index j, that is between [0, size) and is different than i
    :param int i: an index
    :param int size: the size of the matrix
    :returns int j
    """
    size_minus_one = size - 1
    if i == 0:
        return random.randrange(start=1, stop=size)  # any node j other than the first (0)
    elif i == size_minus_one:
        return random.randrange(start=0, stop=size_minus_one)  # any node j except than the last (size-1)
    elif 0 < i < size_minus_one:
        return excluding_randrange(start=0, stop=i, start_again=(i + 1), stop_again=size)