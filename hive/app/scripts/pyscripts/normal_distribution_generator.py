import logging
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def generate_samples(surveys: int = 10,
                     sample_count: int = 10000,
                     mean: float = 34,
                     std: float = 30
                     ) -> np.array:
    """Generates samples from a skewed normal distribution.

    Note:
        If you use this sample generation, simply select pick up the elements
        and assign them to a label in sequence. In this case sample_count is
        just the number of samples I wish to take. See also
        :py:func:`~generate_samples_extended`.

    Args:
        surveys:
            optional; The number of studies to be performed (default is 10).
        sample_count:
            optional; The number of answers in each study. (default is 10000).
        mean:
             optional; Where peak density will be found (default is 34.0).
        std:
            optional; The standard deviation for the plotted normal
            distribution (default is 30.0).

    Returns:
        The sampled bins.
    """
    results = np.zeros((sample_count, surveys))
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            results[i, j] = mean + np.random.normal() * std
    return results


def generate_samples_extended(bin_count: int = 7001,
                              sample_count: int = 7001
                              ) -> Tuple[np.array, np.array]:
    """Generates samples from a normal distribution.

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

    Returns:
        The sampled bins and respective frequencies.
    """
    results: np.array = generate_samples(sample_count)
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
        logging.warning("probability_compensation: " + str(1.0 - total_probability))

    return results, bin_probability


def plot_uptime_distribution(bin_count: Union[int, str] = 'auto',
                             mean: float = 34.0,
                             std: float = 30.0) -> None:
    """Creates and draws a plot of the generated distribution

    Args:
        bin_count:
            optional; The number of bins the plot should have (default is
            'auto'), i.e., matplotlib.pyplot module's functions chooses
            a probably adequate bin count.
        mean:
             optional; Where peak density will be found (default is 34.0).
        std:
            optional; The standard deviation for the plotted normal
            distribution (default is 30.0).
    """
    results: np.array = generate_samples(mean=mean, std=std)
    plt.hist(results, bin_count, density=True)
    plt.title("Peer Uptime Distribution")
    plt.xlabel("Time Spent Online")
    plt.ylabel("Frequency")
    plt.xlim(0.0, 100.0)
    plt.show()


if __name__ == "__main__":
    samples: np.array = generate_samples(mean=float(input("mean: ")), std=float(input("standard deviation: ")))
    plt.hist(samples, 'auto', density=True)
    plt.title("Peer Uptime Distribution")
    plt.xlabel("Time Spent Online")
    plt.ylabel("Frequency")
    plt.xlim(0.0, 100.0)
    plt.show()
