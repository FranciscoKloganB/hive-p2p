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
        :py:func:`generate_samples_extended`.

    Args:
        surveys:
             The number of studies to be performed.
        sample_count:
             The number of answers in each study.
        mean:
              Where peak density will be found.
        std:
             The standard deviation for the plotted normal
            distribution.

    Returns:
        The sampled bins.
    """
    results = np.zeros((sample_count, surveys))
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            results[i, j] = mean + np.random.normal() * std
    return results


def plot_uptime_distribution(bin_count: Union[int, str] = 'auto',
                             mean: float = 34.0,
                             std: float = 30.0) -> None:
    """Creates and draws a plot of the generated distribution

    Args:
        bin_count:
             The number of bins the plot should have. With default
            matplotlib.pyplot module's functions chooses a probably adequate
            bin count.
        mean:
              Where peak density will be found.
        std:
             The standard deviation for the plotted normal
            distribution.
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
