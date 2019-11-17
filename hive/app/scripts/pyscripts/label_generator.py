import string
import itertools
import logging


def yield_label() -> str:
    """
    The yield statement suspends function’s execution and sends a value back to caller, but retains enough state to
    enable function to resume where it is left off. When resumed, the function continues execution immediately after
    the last yield run. This allows its code to produce a series of values over time, rather them computing them at
    once and sending them back like a list.
    :returns str s: a label that follows s-1 in the label sequence.
    """
    for size in itertools.count(1):
        for s in itertools.product(string.ascii_lowercase, repeat=size):
            yield "".join(s)


def __generate_sequence(n: int) -> str:
    """
    This function is just to demonstrate how yield_label() can be used in your code. Do not use it as is.
    :param n: number of desired labels
    """
    logging.warning("This __generate_sequence is a demonstration of how yield_label() works, don't use it production!")
    for s in itertools.islice(yield_label(), n):
        return s
