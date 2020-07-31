import itertools
import string


def yield_label() -> str:
    """Used to generate an arbrirary number of unique labels.
    Yields:
        The next string label in the sequence.

    Examples:
        >>> n = 4
        >>> for s in itertools.islice(yield_label(), n):
        ...     return s
        [a, b, c, d]

       >>> n = 4 + 26
        >>> for s in itertools.islice(yield_label(), n):
        ...     return s
        [a, b, c, d, ..., aa, ab, ac, ad]
    """
    for size in itertools.count(1):
        for s in itertools.product(string.ascii_lowercase, repeat=size):
            yield "".join(s)
