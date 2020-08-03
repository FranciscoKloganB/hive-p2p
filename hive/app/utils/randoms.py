"""This module implements some functions related with random number generation."""

import logging
import random

import numpy as np


def excluding_randrange(start, stop, start_again, stop_again, step=1):
    """Generates a random number

    Args:
        start:
            Number consideration for generation starts from this.
        stop:
            Numbers less than this are generated unless they are bigger or
            equal than `start_again`.
        start_again:
            Number consideration for generation starts again from this.
        stop_again:
            Number consideration stops here and does not include the inputed
            value.
        step:
            optional; Step point of range, this won't be included. This is
            optional (default is 1).
    The random number that is generated is

    Returns:
        A randomly selected element from in the interval [start, stop) or in
        [start_again, stop_again).
    """
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
    """Generates a random number that can be used as a iterables' index.

    Args:
        i:
            An index;
        size:
            The size of the matrix
    Returns:
        A random index that is different than `i` and belongs to [0, `size`).
    """
    size_minus_one = size - 1
    if i == 0:
        # any node j other than the first (0)
        return random.randrange(start=1, stop=size)
    elif i == size_minus_one:
        # any node j except than the last (size-1)
        return random.randrange(start=0, stop=size_minus_one)
    elif 0 < i < size_minus_one:
        return excluding_randrange(
            start=0, stop=i, start_again=(i + 1), stop_again=size)
