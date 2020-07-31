import logging
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skewnorm


def generate_skewed_samples(
        sample_count: int = 10000, skewness: float = -90.0) -> np.array:
    """Generates samples from a skewed normal distribution.

    Note:
        If you use this sample generation, simply select pick up the elements
        and assign them to a label in sequence. In this case sample_count is
        just the number of samples I wish to take. See also
        :py:func:`~generate_skewed_samples_extended`.

    Args:
        sample_count:
            optional; The number of skewed results to be drawn. (default is
            10000).
        skewness:
            optional; Where peak density will be found. Negative values make
             the plot left skewed, positive values make it right skewed (
             default is -90.0).

    Returns:
        The sampled bins.
    """
    # Skew norm function
    results = skewnorm.rvs(a=skewness, size=sample_count)
    # Shift the set so the minimum value is equal to zero
    results = results - min(results)
    # Normalize all the values to be between 0 and 1.
    results = results / max(results)
    # Multiply the standardized values by the maximum value.
    results = results * 100.0

    return results


def generate_skewed_samples_extended(bin_count: int = 7001,
                                     sample_count: int = 10000,
                                     skewness: float = -90.0
                                     ) -> Tuple[np.array, np.array]:
    """Generates samples from a skewed normal distribution.

    Notes:
        A `bin_count` of 7001 represents all values between [30.00, 100.00]
        with 0.01 step, whereas a `bin_count` of 800001 would represents all
        values between [20.0000, 100.000] with 0.0001 step. To let
        matplotlib.skewnorm module define an automatic number of bins use
        bin_count='auto'.

    Args:
        bin_count:
            optional; (default is 7001).
        sample_count:
            optional; The number of skewed results to be drawn. (default is
            10000).
        skewness:
            optional; Where peak density will be found. Negative values make
             the plot left skewed, positive values make it right skewed (
             default is -90.0).

    Returns:
        The sampled bins and respective frequencies.
    """
    results: np.array = generate_skewed_samples(sample_count, skewness)
    bin_density, bins, patches = plt.hist(results, bins=bin_count, density=True)

    size: int = len(bin_density)

    total_density: float = 0.0
    for i in range(size):
        total_density += bin_density[i]

    total_probability: float = 0.0
    bin_probability = bin_density
    for i in range(size):
        bin_probability[i] = bin_density[i] / total_density
        total_probability += bin_probability[i]

    if total_probability != 1.0:
        compensation = str(1.0 - total_probability)
        logging.warning(f"Bin probability is missing to be equal to one: "
                        f"{compensation}.")

    return results, bin_probability


def plot_uptime_distribution(
        bin_count: Union[int, str] = 'auto', skewness: float = -90.0) -> None:
    """Creates and draws a plot of the generated distribution

    Args:
        bin_count:
            optional; The number of bins the plot should have (default is
            'auto'), i.e., matplotlib.pyplot module's functions chooses
            a probably adequate bin count.
        skewness:
             optional; Where peak density will be found. Negative values make
             the plot left skewed, positive values make it right skewed (
             default is -90.0).
    """
    results: np.array = generate_skewed_samples(skewness=skewness)
    plt.hist(results, bin_count, density=True)
    plt.title("Peer Uptime Distribution")
    plt.xlabel("Time Spent Online")
    plt.ylabel("Frequency")
    plt.xlim(0.0, 100.0)
    plt.show()


if __name__ == "__main__":
    samples: np.array = generate_skewed_samples(
        skewness=float(input("input skewness: ")))

    plt.hist(samples, 'auto', density=True)
    plt.title("Peer Uptime Distribution")
    plt.xlabel("Time Spent Online")
    plt.ylabel("Frequency")
    plt.xlim(0.0, 100.0)
    plt.show()
