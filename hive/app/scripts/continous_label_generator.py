import itertools
from string import ascii_lowercase


def yield_label():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            print(s)
            yield "".join(s)


def __generate_sequence(n):
    """
    This function is just to demonstrate how yield label works
    """
    for s in itertools.islice(yield_label(), n):
        return s
