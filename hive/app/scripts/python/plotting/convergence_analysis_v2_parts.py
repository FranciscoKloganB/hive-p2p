"""
This script collects data
"""
import os
import sys
import json
import math
import getopt
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
import _matplotlib_configs as cfg

from matplotlib import rc
from typing import List, Tuple, Any


# region Old Plots (ACC 1.0 Paper) - Trashy Trash
def plotvalues(convergence_times_list, directory, state):
    print()
    # Format data sources
    time_in_convergence = []
    termination_epochs = []
    largest_window = []
    smallest_window = []
    for e in convergence_times_list:
        time_in_convergence.append(e[0])
        termination_epochs.append(e[1])
        largest_window.append(e[3])
        smallest_window.append(e[4])

    # Init figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 7)

    # Format figure bar locations and groups
    width = 0.2  # the width of the bars
    simulation_instance_count = len(convergence_times_list)
    simulation_labels = ["S{}".format(i) for i in range(1, simulation_instance_count + 1)]  # label of each bar
    x = np.arange(simulation_instance_count)  # number of bars
    ax.bar(x - (3/2) * width, time_in_convergence, width, label='time in converrgence', color='darkslategrey')
    ax.bar(x - width / 2, termination_epochs, width, label='termination epoch', color='tan')
    ax.bar(x + width / 2, largest_window, width, label='largest convergence window', color='olivedrab')
    ax.bar(x + (3/2) * width, smallest_window, width, label='smallest convergence window', color='yellowgreen')
    # Set labels
    # ax.set_title("Convergence Analysis - {}i{}".format(directory, state))
    ax.set_xlabel("Simulation Instances")
    ax.set_ylabel("Epochs")
    # Build figure
    ax.set_xticks(x)
    ax.set_xticklabels(simulation_labels)
    plt.axhline(y=np.mean(time_in_convergence),  label="avg. time in convergence", color='darkcyan', linestyle='--')
    plt.axhline(y=np.mean(termination_epochs),  label="avg. termination epoch", color='darkkhaki', linestyle='--')
    # Format legend
    leg = ax.legend(loc='lower center', prop={'size': 9}, ncol=6, fancybox=True, shadow=True)
    # Get the bounding box of the original legend and shift its place
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    bb.y0 -= 0.15  # yOffset
    bb.y1 -= 0.15  # yOffset

    leg.set_bbox_to_anchor(bb, transform=ax.transAxes)

    __save_figure__("CSets", image_ext)


def process_file(filepath, convergence_times_list):
    time_in_convergence = 0
    with open(filepath) as instance:
        # Serialize json file
        json_obj = json.load(instance)
        terminated = json_obj["terminated"]
        largest_convergence_window = json_obj["largest_convergence_window"]
        data = json_obj["convergence_sets"]
        # Calculate how much time the cluster was in convergence
        smallest_convergence_window = terminated

        for convergence_set in data:
            time_in_convergence += (2 + len(convergence_set))
            if len(convergence_set) < smallest_convergence_window:
                smallest_convergence_window = len(convergence_set)

        if smallest_convergence_window == terminated:
            smallest_convergence_window = 0
        else:
            smallest_convergence_window += 2

        convergence_times_list.append(
            (time_in_convergence, terminated, time_in_convergence / terminated, largest_convergence_window, smallest_convergence_window)
        )


def main(directory, state):
    path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'static', 'outfiles')), directory, state)
    convergence_times_list: List[Tuple[int, int, float, int, int]] = []
    for filename in os.listdir(path):
        process_file(os.path.join(path, filename), convergence_times_list)
    # Calculate the global mean at epoch i; Since we have a sum of means, at each epoch, we only need to divide each element by the number of seen instances
    plotvalues(convergence_times_list, directory, state)
# endregion


# region Helpers
def __makedirs__():
    if not os.path.exists(plots_directory):
        os.mkdir(plots_directory)


def __shorten_labels__(labels: List[str]) -> List[str]:
    """Shortens functions' names for better plot labeling.

    Args:
        labels:
            A collection of labels to be shortened.
    """
    blacklist = {"new_", "_transition_matrix"}
    labels_count = len(labels)
    for i in range(labels_count):
        text = labels[i]
        for word in blacklist:
            text = text.replace(word, "")
        labels[i] = text
    return labels


def __set_box_color__(bp: Any, color: str) -> None:
    """Changes the colors of a boxplot.

    Args:
        bp:
            The boxplot reference object to be modified.
        color:
            A string specifying the color to apply to the boxplot in
            hexadecimal RBG.
    """
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def __save_figure__(figname: str, ext: str = "pdf") -> None:
    fname = f"{plots_directory}/{figname}-{ns}_O{opt}_Pde{pde}_Pml{pml}.{ext}"
    plt.savefig(fname, bbox_inches="tight", format=ext)


def __boxplot_and_save__(samples: List[Any], figname: str) -> None:
    plt.figure()
    plt.boxplot(samples, flierprops=cfg.outlyer_shape, whis=0.75, notch=True)
    plt.suptitle("Clusters' distance to the select equilibrium",
                 fontproperties=cfg.fp_title, y=0.995)
    plt.title(subtitle, fontproperties=cfg.fp_subtitle)
    plt.xticks([1], [''])
    plt.ylim(0, 1)
    plt.ylabel(r"distance magnitude / cluster size",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    __save_figure__(figname, image_ext)
# endregion


def barchart_instantaneous_convergence_vs_progress(bucket_size: int = 5) -> None:
    for src in source_keys:
        outfiles_view = sources_files[src]
        epoch_cc = {i: 0 for i in range(1, epochs + 1)}
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                sets = outdata["convergence_sets"]
                for s in sets:
                    for e in s:
                        epoch_cc[e] += 1

        # region create buckets of 5% and allocate the buuckets' values
        bucket_count = int(100 / bucket_size)
        epoch_buckets = [i * bucket_size for i in range(1, bucket_count + 1)]
        epoch_vals = [0] * bucket_count
        for i in range(bucket_count):
            low = bucket_size * i
            high = bucket_size * (i + 1)
            for epoch, count in epoch_cc.items():
                if low < epoch <= high:
                    epoch_vals[i] += count
        # endregion

        plt.figure()
        plt.bar(epoch_buckets, epoch_vals, width=bucket_size*.6)
        plt.axhline(y=np.mean(epoch_vals), color='c', linestyle='--')
        # plt.suptitle(
        #     "Number of convergences as simulations' progress",
        #     fontproperties=cfg.fp_title, y=0.995
        # )
        # plt.title(f"{src}, {subtitle}", fontproperties=cfg.fp_subtitle)
        plt.title(f"FBR: {src}, {subtitle}", fontproperties=cfg.fp_title)
        plt.xlabel("simulations' progress (%)",
                   labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
        plt.xlim(bucket_size - bucket_size * 0.5, 100 + bucket_size*0.5)
        plt.ylabel(r"c$_{t}$ count",
                   labelpad=cfg.labels_pad,
                   fontproperties=cfg.fp_axis_labels)
        plt.ylim(0, len(outfiles_view) * bucket_size + 20)
        __save_figure__(f"ICC{src}", image_ext)


def boxplot_first_convergence():
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                csets = outdata["convergence_sets"]
                if csets:
                    data_dict[src_key].append(csets[0][0])

    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values(), flierprops=cfg.outlyer_shape, whis=0.75,
               notch=False)
    ax.set_xticklabels(data_dict.keys())
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    plt.title("clusters' first instantaneous convergence", fontproperties=cfg.fp_title)
    plt.xlabel("number of replicas", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel("epoch", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    __save_figure__("FIC", image_ext)


def piechart_avg_convergence_achieved(single_fig: bool = False) -> None:
    for src in source_keys:
        outfiles_view = sources_files[src]
        data = [0.0, 0.0]
        labels = ["achieved", "has not achieved"]
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                classifications = outdata["topologies_goal_achieved"]
                for success in classifications:
                    data[0 if success else 1] += 1

        fig, ax = plt.subplots()
        ax.axis('equal')
        plt.suptitle("Clusters (%) achieving the selected equilibrium",
                     fontproperties=cfg.fp_title, y=0.995)
        plt.title(f"FBR: {src}, {subtitle}", fontproperties=cfg.fp_subtitle)
        wedges, _, _ = ax.pie(data, startangle=90, autopct='%1.1f%%',
                              labels=labels, labeldistance=None,
                              textprops={'color': 'white', 'weight': 'bold'})
        # bbox_to_anchor(Xanchor, Yanchor, Xc_offset,  Yc_offset)
        # axis 'equal' ensures that pie is drawn as a circle.
        leg = ax.legend(wedges,
                        labels,
                        frameon=False,
                        loc="center left",
                        bbox_to_anchor=(0.7, 0.1, 0, 0))
        # leg.set_title("achieved goal", prop=cfg.fp_axis_labels)
        # leg._legend_box.sep = cfg.legends_pad
        __save_figure__(f"GA{src}", image_ext)


def boxplot_percent_time_instantaneous_convergence():
    samples = []
    outfiles_view = []
    for filename in outfiles_view:
        filepath = os.path.join(directory, filename)
        with open(filepath) as outfile:
            time_in_convergence = 0
            outdata = json.load(outfile)
            sets = outdata["convergence_sets"]
            for s in sets:
                time_in_convergence += len(s)
            samples.append(time_in_convergence / outdata["terminated"])

    plt.figure()
    plt.boxplot(samples, flierprops=cfg.outlyer_shape, whis=0.75, notch=True)
    plt.suptitle("Clusters' time spent in convergence", fontproperties=cfg.fp_title, y=0.995)
    plt.title(subtitle, fontproperties=cfg.fp_subtitle)
    plt.xticks([1], [''])
    plt.ylim(0, 1)
    plt.ylabel(r"sum(c$_{t}$) / termination epoch",
               labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    __save_figure__("TSIC", image_ext)


def boxplot_avg_convergence_magnitude_distance():
    psamples = []
    nsamples = []
    outfiles_view = []
    for filename in outfiles_view:
        filepath = os.path.join(directory, filename)
        with open(filepath) as outfile:
            outdata = json.load(outfile)
            classifications = outdata["topologies_goal_achieved"]
            magnitudes = outdata["topologies_goal_distance"]
            for success, mag in zip_longest(classifications, magnitudes):
                normalized_mag = mag / outdata["original_size"]
                (psamples if success else nsamples).append(normalized_mag)

    __boxplot_and_save__(psamples, "MDS")
    __boxplot_and_save__(nsamples, "MDNS")


if __name__ == "__main__":
    # region args processing
    ns = 8
    epochs = 0
    pde = 0.0
    pml = 0.0
    opt = "off"
    image_ext = "pdf"

    short_opts = "e:o:i:"
    long_opts = ["epochs=", "optimizations=", "image_format="]

    try:
        options, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for options, args in options:
            if options in ("-e", "--epochs"):
                epochs = int(str(args).strip())
            if options in ("-o", "--optimizations"):
                opt = str(args).strip()
            if options in ("-i", "--image_format"):
                image_ext = str(args).strip()

    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --epochs -e (int)\n"
                 "  --optimizations -o (str)\n")
    # endregion

    # region path setup
    directory = os.path.abspath(
        os.path.join(os.getcwd(), '..', '..', '..', 'static', 'outfiles'))

    plots_directory = os.path.join(
        directory, 'simulation_plots', 'convergence_analysis')

    __makedirs__()
    # endregion

    # region sample src setup
    dirfiles = list(
        filter(lambda f: f.endswith(".json"), os.listdir(directory)))

    with open(os.path.join(directory, dirfiles[-1])) as outfile:
        outdata = json.load(outfile)
        ns = outdata["original_size"]
        pde = outdata["corruption_chance_tod"]
        pml = outdata["channel_loss"]

    subtitle = f"Cluster size: {ns}, Opt.: {opt}, P(de): {pde}%, P(ml): {pml}%"

    sources = 5
    sources_files = {
        "-100P": [],
        # "-500P": [],
        "-1000P": [],
        # "-1500P": [],
        "-2000P": []
    }
    for filename in dirfiles:
        for key in sources_files:
            if key in filename:
                sources_files[key].append(filename)
                break

    # remove '-' and 'P' delimiters that avoid incorrect missmapping of sources
    sources_files = {k[1:-1]: v for k, v in sources_files.items()}
    source_keys = list(sources_files)
    # endregion

    # Q2. Existem mais conjuntos de convergencia perto do fim da simulação?
    barchart_instantaneous_convergence_vs_progress(bucket_size=5)
    # Q3. Quanto tempo é preciso até observar a primeira convergencia na rede?
    boxplot_first_convergence()
    # Q4. A média dos vectores de distribuição é proxima ao objetivo?
    piechart_avg_convergence_achieved()
    # boxplot_avg_convergence_magnitude_distance()
    # Q5. Quantas partes são suficientes para um Swarm Guidance  satisfatório?
    # boxplot_percent_time_instantaneous_convergence()
