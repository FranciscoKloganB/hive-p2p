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
from typing import List, Tuple, Any, Dict, Optional


# region Helpers
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


def __save_figure__(figname: str, figext: str = "png") -> None:
    fname = f"{plots_directory}/{figname}-{ns}_O{opt}_Pde{pde}_Pml{pml}.{figext}"
    plt.savefig(fname, bbox_inches="tight", format=figext)


def __create_boxplot__(
        data_dict: Dict[str, Any], suptitle: str, xlabel: str, ylabel: str, figname: str, figext: str = "png", savefig: bool = True
) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values(), flierprops=cfg.outlyer_shape, whis=0.75, notch=True)
    ax.set_xticklabels(data_dict.keys())
    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    if savefig:
        plt.savefig(f"{plots_directory}/{figname}-{ns}_O{opt}_Pde{pde}_Pml{pml}.{figext}",
                    format=figext, bbox_inches="tight")
    return fig, ax


def __create_double_boxplot__(
        left_data, right_data,
        suptitle: str, xlabel: str, ylabel: str, figname: str, figext: str = "png",
        left_color: Optional[str] = None, right_color: Optional[str] = None,
        left_label: Optional[str] = None, right_label: Optional[str] = None,
        savefig: bool = True
) -> Tuple[Any, Any]:

    fig, ax = plt.subplots()
    bpl = plt.boxplot(left_data, sym='', whis=0.75, widths=0.7, notch=True, positions=np.array(range(len(left_data))) * 2.0 - 0.4)
    bpr = plt.boxplot(right_data,  sym='', whis=0.75, widths=0.7, notch=True, positions=np.array(range(len(right_data))) * 2.0 + 0.4)

    if left_color:
        __set_box_color__(bpl, "#55A868")
        if left_label:
            plt.plot([], c="#55A868", label=left_label)
            plt.legend(loc="best", prop=cfg.fp_axis_legend)
    if right_color:
        __set_box_color__(bpr, "#C44E52")
        if right_label:
            plt.plot([], c="#C44E52", label=right_label)
            plt.legend(loc="best", prop=cfg.fp_axis_legend)

    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    if savefig:
        plt.savefig(f"{plots_directory}/{figname}-{ns}_O{opt}_Pde{pde}_Pml{pml}.{figext}",
                    format=figext, bbox_inches="tight")
    return fig, ax
# endregion


# region Boxplots
def boxplot_bandwidth(rl: int = 3, figname: str = "BW") -> None:
    filesize = 47185920  # bytes
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                be = outdata["blocks_existing"][0]
                rl = outdata["replication_level"]
                rl = 3 if rl == 1 else rl  # Hack to compensate mistake in simulationse
                blocksize = ((filesize / be) * rl) / 1024 / 1024  # from B to KB to MB
                c_bandwidth = np.asarray(outdata["blocks_moved"]) * blocksize
                data_dict[src_key].extend(c_bandwidth)

    __create_boxplot__(
        data_dict,
        suptitle="clusters' bandwidth expenditure",
        xlabel="config", ylabel="moved blocks (MB)",
        figname=figname, figext=image_ext)


def boxplot_first_convergence(figname: str = "FIC"):
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                csets = outdata["convergence_sets"]
                if csets:
                    data_dict[src_key].append(csets[0][0])

    __create_boxplot__(
        data_dict,
        suptitle="clusters' first instantaneous convergence",
        xlabel="config", ylabel="epoch",
        figname=figname, figext=image_ext)


def boxplot_percent_time_instantaneous_convergence(figname: str = "TSIC"):
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

    __create_boxplot__(
        data_dict,
        suptitle="clusters' time spent in convergence",
        xlabel="config", ylabel=r"sum(c$_{t}$) / termination epoch",
        figname=figname, figext=image_ext)


def boxplot_avg_convergence_magnitude_distance(figname: str = "MD"):
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

    __create_double_boxplot__(
        psamples, nsamples,
        left_color="#55A868", right_color="#C44E52",
        left_label="achieved eq.", right_label="has not achieved eq.",
        suptitle="clusters' distance to the select equilibrium",
        xlabel="config", ylabel=r"c$_{dm}$ / cluster size",
        figname=figname, figext=image_ext, savefig=False)

    plt.xticks(range(0, len(source_keys) * 2, 2), source_keys, rotation=45, fontsize="x-large", fontweight="semibold")
    plt.xlim(-2, len(source_keys) * 2)
    __save_figure__(figname, image_ext)
# endregion


def barchart_instantaneous_convergence_vs_progress(
        bucket_size: int = 5, figname: str = "ICC") -> None:
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
    __save_figure__(figname, image_ext)


def piechart_avg_convergence_achieved(figname: str = "GA") -> None:
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
    __save_figure__(figname, image_ext)


def setup_sources(source_patterns: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    sources_files = {k: [] for k in source_patterns}
    for filename in dirfiles:
        for key in sources_files:
            if key in filename:
                sources_files[key].append(filename)
                break

    # remove '-' and 'P' delimiters that avoid incorrect missmapping of sources
    # sources_files = {k[1:-1]: v for k, v in sources_files.items()}
    source_keys = list(sources_files)
    return sources_files, source_keys


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

    if not os.path.exists(plots_directory):
        os.mkdir(plots_directory)
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

    source_patterns = ["SG8-100P", "SG8-1000P", "SG8-2000P"]
    sources_files, source_keys = setup_sources(source_patterns)
    # endregion

    # Q1. Quantas mensagens passam na rede por epoch?
    boxplot_bandwidth()
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
    source_patterns = ["SG8-1000P", "SG8-Opt"]
    sources_files, source_keys = setup_sources(source_patterns)
    boxplot_bandwidth()
    boxplot_first_convergence()
    boxplot_avg_convergence_magnitude_distance()
    boxplot_percent_time_instantaneous_convergence()
    # TODO: alter source file dicts, subtitle and recall functions with diff params
