"""Creates a bar chart that visually demonstrates the performance difference
between Hives and PeerSim P2P simulators, both running 30 cycles of the same
two protocols. Newscast with Peer-Shuffling and Network Degree aggregation.
Experiments differ in the number of network nodes that exist in the
simulation."""
from __future__ import annotations

import numpy

import matplotlib.pyplot as plt
import _matplotlib_configs as cfg


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate("{}".format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom")


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
    rects1 = ax.bar(x - width/2, peersim_times, width, label="PeerSim")
    rects2 = ax.bar(x + width/2, hives_times, width, label="Hives")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Use supertitle as title and title as subtitle
    plt.suptitle("Simulators' performance comparison",
                 fontproperties=cfg.fp_title,
                 x=0.56, y=0.999)
    plt.title("30 cycles of Newscast Shuffling and AverageFunction aggregation",
              fontproperties=cfg.fp_subtitle)
    plt.xlabel("Number of network nodes",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    plt.ylabel("Execution time in seconds",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)

    autolabel(rects1)
    autolabel(rects2)

    plt.ylim(0, 250)

    fig.tight_layout()

    plt.show()
