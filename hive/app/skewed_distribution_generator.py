import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skewnorm
from globals import DEBUG


def generate_skewed_samples(sample_count=20000, skewness=-90.0):
    """
    Generates samples from a skewed normal distribution.
    Note:
     If you use this sample generation, simply select pick up the elements and assign them to a label in sequence.
     In this case sample_count is just the number of samples I wish to take. See difference w.r.t extendend version
    :param sample_count: the number of skewed samples to be drawn
    :type int
    :param skewness: where peak density will be found. (-) values are left skewed, (+) values are right skewed
    :type float
    :return samples: drawn from skewed normal distribution
    :type 1-D numpy.array
    """
    samples = skewnorm.rvs(a=skewness, size=sample_count)  # Skewnorm function
    samples = samples - min(samples)  # Shift the set so the minimum value is equal to zero
    samples = samples / max(samples)  # Standadize all the vlues between 0 and 1.
    samples = samples * 100.0         # Multiply the standardized values by the maximum value.
    return samples


def generate_skwed_samples_extended(bin_count=7001, sample_count=7001, skewness=-90.0):
    """
    Generates samples from a skewed normal distribution.
    Note:
     bin_count at 7001 represents all values between [30.00, 100.00] with 0.01 step
     bin_count at 800001 would represents all values between [20.0000, 100.000] with 0.0001 step, and so on...
     To let matplotlib.skewnorm module define an automatic number of bins use bin_count='auto'
     Keeping bins_count = sample_count is just an hack to facilitate np.random.choice(bins_count, sample_count)
     Because we are using bin_counts here, it is advised not to draw to many samples, as the function will be exponen-
     tially slower
    :param bin_count: the number of bins to be created in the matplotlib.pyplot.hist.bins
    :type int
    :param sample_count: the number of skewed samples to be drawn
    :type int
    :param skewness: where peak density will be found. (-) values are left skewed, (+) values are right skewed
    :type float
    :returns samples, bin_probability: drawn from skewed normal distribution, probability of each sample ocurring
    :type tuple<1-D numpy.array, 1-D numpy.array>
    """
    samples = generate_skewed_samples(sample_count, skewness)
    bin_density, bins, patches = plt.hist(samples, bins=bin_count, density=True)

    if DEBUG:
        plt.show()
        print("total_density (numpy): " + str(np.sum(bin_density)))

    size = len(bin_density)

    total_density = 0.0
    for i in range(size):
        total_density += bin_density[i]

    total_probability = 0.0
    bin_probability = bin_density
    for i in range(size):
        bin_probability[i] = bin_density[i] / total_density
        total_probability += bin_probability[i]

    if total_probability != 1.0:
        logging.warning("probability_compensation: " + str(1.0 - total_probability))

    if DEBUG:
        print("total_density (for loop): " + str(total_density))
        print("total_probability (numpy): " + str(np.sum(bin_probability)))
        print("total_probability (for loop): " + str(total_probability))
        print("number_of_bins: " + str(len(bin_density)))
        print("number_of_samples: " + str(len(samples)))

    return samples, bin_probability


def plot_uptime_distribution(bin_count='auto', sample_count=10000, skewness=-90.0):
    """
    Displays generate_skewed_samples in a two axis plot
    :param bin_count: the number of bins to be depicted in the matplotlib.pyplot.hist plot
    :type int
    :param sample_count: the number of skewed samples to be drawn
    :type int
    :param skewness: where peak density will be found. (-) values are left skewed, (+) values are right skewed
    :type float
    """
    samples = generate_skewed_samples(sample_count, skewness)
    plt.hist(samples, bin_count, density=True)
    plt.title("Peer Node Uptime Distribution")
    plt.xlabel("uptime")
    plt.ylabel("density")
    plt.show()
