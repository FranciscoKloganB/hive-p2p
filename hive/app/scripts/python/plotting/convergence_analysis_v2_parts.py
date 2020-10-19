"""
This script collects data
"""
import os
import sys
import json
import math
import getopt
import operator
import functools
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
import _matplotlib_configs as cfg

from matplotlib import rc
from typing import List, Tuple, Any, Dict


# region Helpers
def __makedirs__():
    if not os.path.exists(plots_directory):
        os.mkdir(plots_directory)


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


def __create_boxplot__(
        data_dict: Dict[str, Any], suptitle: str, xlabel: str, ylabel: str,
) -> None:
    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values(), flierprops=cfg.outlyer_shape, whis=0.75, notch=True)
    ax.set_xticklabels(data_dict.keys())
    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
# endregion


def boxplot_bandwidth(rl: int = 3, image_name: str = "BW") -> None:
    filesize = 47185920  # bytes
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                be = outdata["blocks_existing"]
                rl = outdata["replication_level"]
                rl = 3 if rl == 1 else rl  # Hack to compensate mistake in simulations
                blocksize = (filesize / be) * rl
                c_bandwidth = np.asarray(outdata["blocks_moved"]) * blocksize
                data_dict[src_key].append(c_bandwidth)

    __create_boxplot__(data_dict,
                       suptitle="clusters' bandwidth expenditure",
                       xlabel="configuration", ylabel="moved blocks x read size")



def barchart_instantaneous_convergence_vs_progress(
        bucket_size: int = 5, image_name: str = "ICC") -> None:
    # region create buckets of 5%
    bucket_count = int(100 / bucket_size)
    epoch_buckets = [i * bucket_size for i in range(1, bucket_count + 1)]
    # endregion

    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
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
        epoch_vals = [0] * bucket_count
        for i in range(bucket_count):
            low = bucket_size * i
            high = bucket_size * (i + 1)
            for epoch, count in epoch_cc.items():
                if low < epoch <= high:
                    epoch_vals[i] += count
        data_dict[src_key] = epoch_vals
        # endregion

    bar_locations = np.asarray(epoch_buckets)
    bar_width = bucket_size * 0.25

    fig, ax = plt.subplots()
    plt.suptitle("convergence observations as simulations' progress", fontproperties=cfg.fp_title)
    plt.xlabel("simulations' progress (%)", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(r"c$_{t}$ count", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xlim(bucket_size - bucket_size * 0.75, 100 + bucket_size*0.8)
    plt.ylim(0, 100)
    plt.xticks(rotation=75, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    ax.set_xticks(epoch_buckets)
    for i in range(len(source_keys)):
        key = source_keys[i]
        epoch_vals = data_dict[key]
        ax.bar(bar_locations + (bar_width * i) - 0.5 * bar_width, epoch_vals, width=bar_width)
    ax.legend([f"{x} parts" for x in source_keys], prop=cfg.fp_axis_legend)
    __save_figure__(image_name, image_ext)


def boxplot_first_convergence(image_name: str = "FIC"):
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
    ax.boxplot(data_dict.values(), flierprops=cfg.outlyer_shape, whis=0.75, notch=True)
    ax.set_xticklabels(data_dict.keys())
    plt.suptitle("clusters' first instantaneous convergence", fontproperties=cfg.fp_title)
    plt.xlabel("number of replicas", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel("epoch", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    __save_figure__(image_name, image_ext)


def piechart_avg_convergence_achieved(image_name: str = "GA") -> None:
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        data = [0.0, 0.0]
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                classifications = outdata["topologies_goal_achieved"]
                for success in classifications:
                    data[0 if success else 1] += 1
        data_dict[src_key] = list(data)

    s = len(source_keys)
    wedge_labels = ["achieved eq.", "has not achieved eq."]
    fig, axes = plt.subplots(1, s, figsize=(s*3, s))
    plt.suptitle("clusters (%) achieving the selected equilibrium on average", fontproperties=cfg.fp_title, x=0.51)
    for i, ax in enumerate(axes.flatten()):
        ax.axis('equal')
        src_key = source_keys[i]
        wedges, _, _ = ax.pie(
            data_dict[src_key],
            startangle=90, autopct='%1.1f%%',
            labels=wedge_labels, labeldistance=None,
            textprops={'color': 'white', 'weight': 'bold'}
        )
        ax.set_xlabel(f"{src_key} parts", fontproperties=cfg.fp_axis_legend)
    plt.legend(labels=wedge_labels, ncol=s, frameon=False,
               loc="best", bbox_to_anchor=(0.5, -0.2),
               prop=cfg.fp_axis_legend)
    __save_figure__(image_name, image_ext)


def boxplot_percent_time_instantaneous_convergence(image_name: str = "TSIC"):
    # region create data samples for each source
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                time_in_convergence = 0
                outdata = json.load(outfile)
                for s in outdata["convergence_sets"]:
                    time_in_convergence += len(s)
                data_dict[src_key].append(time_in_convergence / outdata["terminated"])

    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values(), flierprops=cfg.outlyer_shape, whis=0.75, notch=True)
    ax.set_xticklabels(data_dict.keys())
    plt.suptitle("clusters' time spent in convergence", fontproperties=cfg.fp_title)
    plt.xlabel("number of replicas", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(r"sum(c$_{t}$) / termination epoch", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    plt.ylim(0, 1)
    __save_figure__(image_name, image_ext)


def boxplot_avg_convergence_magnitude_distance(image_name: str = "MD"):
    # region create data samples for each source
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        psamples = []
        nsamples = []
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                classifications = outdata["topologies_goal_achieved"]
                magnitudes = outdata["topologies_goal_distance"]
                for success, mag in zip_longest(classifications, magnitudes):
                    normalized_mag = mag / outdata["original_size"]
                    (psamples if success else nsamples).append(normalized_mag)
        data_dict[src_key] = [psamples, nsamples]
    # endregion

    psamples = []
    nsamples = []
    for src_key in source_keys:
        psamples.append(data_dict[src_key][0])
        nsamples.append(data_dict[src_key][1])

    plt.figure()
    plt.suptitle("clusters' distance to the select equilibrium", fontproperties=cfg.fp_title)
    plt.ylabel(r"c$_{dm}$ / cluster size", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xlabel("number of parts in the cluster", labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)

    bpleft = plt.boxplot(psamples, sym='', whis=0.75, widths=0.7, notch=True,
                         positions=np.array(range(len(psamples))) * 2.0 - 0.4)
    bpright = plt.boxplot(nsamples,  sym='', whis=0.75, widths=0.7, notch=True,
                          positions=np.array(range(len(nsamples))) * 2.0 + 0.4)

    __set_box_color__(bpleft, "#55A868")  # colors are from http://colorbrewer2.org/
    __set_box_color__(bpright, "#C44E52")

    # craete two fake empty plots for easy labeling
    plt.plot([], c="#55A868", label='achieved eq.')
    plt.plot([], c="#C44E52", label='has not achieved eq.')
    plt.legend(loc="best", prop=cfg.fp_axis_legend)

    plt.xticks(range(0, len(source_keys) * 2, 2), source_keys, rotation=75, fontsize="x-large", fontweight="semibold")
    plt.xlim(-2, len(source_keys) * 2)
    plt.ylim(0, 0.3)
    plt.yticks(fontsize="x-large", fontweight="semibold")

    __save_figure__(image_name, image_ext)


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

    # Q1. Quantas mensagens passam na rede por epoch?
    # TODO: create boxplot and plots
    # Q2. Existem mais conjuntos de convergencia perto do fim da simulação?
    barchart_instantaneous_convergence_vs_progress(bucket_size=5)
    # Q3. Quanto tempo é preciso até observar a primeira convergencia na rede?
    boxplot_first_convergence()
    # Q4. A média dos vectores de distribuição é proxima ao objetivo?
    piechart_avg_convergence_achieved()
    boxplot_avg_convergence_magnitude_distance()
    # Q5. Quantas partes são suficientes para um Swarm Guidance  satisfatório?
    boxplot_percent_time_instantaneous_convergence()
    # Q6. Tecnicas de optimização influenciam as questões anteriores?
    # TODO: alter source file dicts, subtitle and recall functions with diff params
