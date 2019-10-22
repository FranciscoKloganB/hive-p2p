import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import skewnorm

DEBUG = False

def generate_skewed_samples(sample_count=10000):
    """
    If you use this sample generation, simply select pick up the elements and assign them to a label in sequence.
    # In this case sample_count is just the number of samples I wish to take. See difference w.r.t extendend version
    """
    max_uptime = 100.0
    skewness = -90.0  # Negative values are left skewed, positive values are right skewed. DON'T REMOVE (-) sign
    samples = skewnorm.rvs(a=skewness, size=sample_count)  # Skewnorm function
    samples = samples - min(samples)  # Shift the set so the minimum value is equal to zero
    samples = samples / max(samples)  # Standadize all the vlues between 0 and 1.
    samples = samples * max_uptime    # Multiply the standardized values by the maximum value.
    return samples


def generate_skwed_samples_extended(bin_count=7001, sample_count=7001):
    """
    If you use this sample generation, use np.random.choice
    # 7001 represents all values between [30.00, 100.00] with 0.01 step
    # 800001 Could represents all values between [20.0000, 100.000] with 0.0001 step, etc
    # To define an auto number of bins use bin='auto'
    # Keep bins_count = sample_count is just an hack to facilitate np.random.choice(bins_count, sample_count)
    """
    samples = generate_skewed_samples(sample_count)
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


def plot_uptime_distribution(bin_count, sample_count):
    samples = generate_skewed_samples(sample_count)
    n, bins, patches = plt.hist(samples, bin_count, density=True)
    plt.title("Peer Node Uptime Distribution")
    plt.xlabel("uptime")
    plt.ylabel("density")
    plt.show()
