import logging

import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Union


def generate_samples(surveys: int = 10, sample_count: int = 10000, mean: float = 34, std: float = 30) -> np.array:
    """
    Generates samples from a skewed normal distribution.
    Note:
     If you use this sample generation, simply select pick up the elements and assign them to a label in sequence.
     In this case sample_count is just the number of samples I wish to take. See difference w.r.t extended version
    :param int surveys: the number of studies performed
    :param int sample_count: the number of answers in each study
    :param float mean: where the mean of the distribution is
    :param float std: standard deviation
    :return np.array samples: drawn from skewed normal distribution
    """
    results = np.zeros((sample_count, surveys))
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            results[i, j] = mean + np.random.normal() * std
    return results


def generate_samples_extended(bin_count: int = 7001, sample_count: int = 7001) -> Tuple[np.array, np.array]:
    """
    Generates samples from a skewed normal distribution.
     surveys at 7001 represents all values between [30.00, 100.00] with 0.01 step
     surveys at 800001 would represents all values between [20.0000, 100.000] with 0.0001 step, and so on...
     To let matplotlib.skewnorm module define an automatic number of bins use surveys='auto'
     Keeping bins_count = sample_count is just an hack to facilitate np.random.choice(bins_count, sample_count)
     Because we are using bin_counts here, it is advised not to draw to many samples, as the function will be
     exponentially slower
    :param int bin_count: the number of bins to be created in the matplotlib.pyplot.hist.bins
    :param int sample_count: the number of skewed samples to be drawn
    :returns Tuple[np.array, np.array] samples, bin_probability: sample and respective probability of occurring
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


def plot_uptime_distribution(bin_count: Union[int, str] = 'auto', mean: float = 34.0, std: float = 33.0) -> None:
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
