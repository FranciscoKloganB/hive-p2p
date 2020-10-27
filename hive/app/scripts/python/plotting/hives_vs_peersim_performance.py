"""Creates a bar chart that visually demonstrates the performance difference
between Hives and PeerSim P2P simulators, both running 30 cycles of the same
two protocols. Newscast with Peer-Shuffling and Network Degree aggregation.
Experiments differ in the number of network nodes that exist in the
simulation."""
from __future__ import annotations

import os

import numpy

import matplotlib.pyplot as plt
from _matplotlib_configs import *


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize="large", fontweight="semibold", color="dimgrey")


if __name__ == "__main__":
    labels = (
        "10\N{SUPERSCRIPT TWO}",
        "10\N{SUPERSCRIPT THREE}",
        "10\N{SUPERSCRIPT FOUR}",
        "10\N{SUPERSCRIPT FIVE}"
    )
    peersim_times = (1, 1, 2, 6)
    hives_times = (1, 2, 21, 232)

    x = numpy.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    rects1 = ax.bar(x - width/2, peersim_times, width, label="PeerSim")
    rects2 = ax.bar(x + width/2, hives_times, width, label="Hives")

    # Use supertitle as title and title as subtitle
    # plt.suptitle("Simulators' performance comparison, Hives vs. PeerSim",
    #              fontproperties=fp_title,
    #              x=0.56, y=0.999)
    # plt.title("30 cycles of Newscast Shuffling and AverageFunction aggregation",
    #           fontproperties=fp_subtitle)
    # plt.title("Simulators' performance comparison, Hives vs. PeerSim",
    #          pad=title_pad, fontproperties=fp_title)

    plt.xlabel("number of network nodes",
               labelpad=labels_pad, fontproperties=fp_axis_labels)

    plt.ylabel("time in seconds",
               labelpad=labels_pad, fontproperties=fp_axis_labels)

    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    autolabel(rects1)
    autolabel(rects2)

    plt.ylim(0, 250)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=2, prop=fp_axis_legend,
              loc="lower center", bbox_to_anchor=(0.5, -0.425))

    figdir = os.path.abspath(os.path.join(
        os.getcwd(), '..', '..', '..', 'static', 'outfiles', 'simulation_plots')
    )
    figname = f"{figdir}/simulators_execution_times.pdf"

    plt.savefig(figname, bbox_inches="tight", format="pdf")
