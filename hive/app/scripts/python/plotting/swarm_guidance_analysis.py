"""
This script collects data
"""
import getopt
import json
import os
import sys
from itertools import zip_longest
from typing import List, Tuple, Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

import _matplotlib_configs as cfg


# region Helpers
def setup_sources(source_patterns: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    sources_files = {k: [] for k in source_patterns}
    for filename in dirfiles:
        for key in sources_files:
            if key in filename:
                sources_files[key].append(filename)
                break
    source_keys = list(sources_files)
    return sources_files, source_keys


def __get_whole_frac__(num: float) -> Tuple[int, int]:
    return int(num), int(str(num)[(len(str(int(num)))+1):])


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
    fname = f"{plots_directory}/{figname}.{figext}"
    plt.savefig(fname, bbox_inches="tight", format=figext)


def __create_barchart__(data_dict: Dict[str, Any],
                        bar_locations: np.ndarray, bar_width: float,
                        bucket_size: float,
                        suptitle: str, xlabel: str, ylabel: str,
                        figname: str, figext: str = "png",
                        savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()
    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=75, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    plt.xlim(bucket_size - bucket_size * 0.75, 100 + bucket_size * 0.8)
    ax.set_xticks(bar_locations)
    for i in range(len(source_keys)):
        key = source_keys[i]
        epoch_vals = data_dict[key]
        ax.bar(bar_locations + (bar_width * i) - 0.5 * bar_width, epoch_vals, width=bar_width)
    ax.legend([str(x) for x in source_keys], frameon=False, loc="best", prop=cfg.fp_axis_legend)

    if savefig:
        plt.savefig(f"{plots_directory}/{figname}.{figext}", format=figext, bbox_inches="tight")

    return fig, ax


def __create_boxplot__(data_dict: Dict[str, Any],
                       suptitle: str, xlabel: str, ylabel: str,
                       figname: str, figext: str = "png",
                       savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()
    ax.boxplot(data_dict.values(), flierprops=cfg.outlyer_shape, whis=0.75, notch=True)
    ax.set_xticklabels(data_dict.keys())
    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=75, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    if savefig:
        plt.savefig(f"{plots_directory}/{figname}.{figext}", format=figext, bbox_inches="tight")
    return fig, ax


def __create_double_boxplot__(left_data, right_data,
                              suptitle: str, xlabel: str, ylabel: str,
                              labels: List[str],
                              figname: str, figext: str = "png",
                              left_color: Optional[str] = None,
                              right_color: Optional[str] = None,
                              left_label: Optional[str] = None,
                              right_label: Optional[str] = None,
                              savefig: bool = True) -> Tuple[Any, Any]:

    fig, ax = plt.subplots()
    bpl = plt.boxplot(left_data, sym='', whis=0.75, widths=0.7, notch=True, positions=np.array(range(len(left_data))) * 2.0 - 0.4)
    bpr = plt.boxplot(right_data,  sym='', whis=0.75, widths=0.7, notch=True, positions=np.array(range(len(right_data))) * 2.0 + 0.4)

    if left_color:
        __set_box_color__(bpl, left_color)
        if left_label:
            plt.plot([], c=left_color, label=left_label)
            plt.legend(frameon=False, loc="best", prop=cfg.fp_axis_legend)
    if right_color:
        __set_box_color__(bpr, right_color)
        if right_label:
            plt.plot([], c=right_color, label=right_label)
            plt.legend(frameon=False, loc="best", prop=cfg.fp_axis_legend)

    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad, fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=75, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    label_count = len(labels)
    plt.xticks(range(0, label_count * 2, 2), labels, rotation=75, fontsize="x-large", fontweight="semibold")
    plt.xlim(-2, label_count * 2)

    if savefig:
        plt.savefig(f"{plots_directory}/{figname}.{figext}", format=figext, bbox_inches="tight")
    return fig, ax
# endregion


# region Boxplots
def boxplot_bandwidth(figname: str = "BW") -> None:
    filesize = 47185920  # bytes
    # region create data dict
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
    # endregion

    __create_boxplot__(
        data_dict,
        suptitle="clusters' bandwidth expenditure",
        xlabel="config", ylabel="moved blocks (MB)",
        figname=figname, figext=image_ext)


def boxplot_first_convergence(figname: str = "FIC") -> None:
    # region create data dict
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                csets = outdata["convergence_sets"]
                if csets:
                    data_dict[src_key].append(csets[0][0])
    # endregion

    __create_boxplot__(
        data_dict,
        suptitle="clusters' first instantaneous convergence",
        xlabel="config", ylabel="epoch",
        figname=figname, figext=image_ext)


def boxplot_percent_time_instantaneous_convergence(figname: str = "TIC") -> None:
    # region create data dict
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
    # endregion

    __create_boxplot__(
        data_dict,
        suptitle="clusters' time spent in convergence",
        xlabel="config", ylabel=r"sum(c$_{t}$) / termination epoch",
        figname=figname, figext=image_ext)


def boxplot_avg_convergence_magnitude_distance(figname: str = "MD") -> None:
    # region create data dict
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
        labels=source_keys,
        figname=figname, figext=image_ext)


def boxplot_node_degree(figname: str = "ND") -> None:
    """The integral part of the float value is the
    in-degree, the decimal part is the out-degree."""
    # region create data dict
    data_dict = {k: [[], []] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                matrices_degrees: List[Dict[str, float]] = outdata["matrices_nodes_degrees"]
                for topology in matrices_degrees:
                    for nodes_degree in topology.values():
                        in_degree, out_degree = __get_whole_frac__(nodes_degree)
                        data_dict[src_key][0].append(in_degree)
                        data_dict[src_key][1].append(out_degree)
    # endregion

    isamples = []
    osamples = []
    for src_key in source_keys:
        isamples.append(data_dict[src_key][0])
        osamples.append(data_dict[src_key][1])

    __create_double_boxplot__(
        isamples, osamples,
        left_color="#4C72B0", right_color="#55A868",
        left_label="in-degree", right_label="out-degree",
        suptitle="Nodes' degrees depending on the cluster's size",
        xlabel="config", ylabel="node degrees",
        labels=source_keys,
        figname=figname, figext=image_ext)

# endregion


# region Bar charts
def barchart_instantaneous_convergence_vs_progress(
        bucket_size: int = 5, figname: str = "ICC") -> None:
    # region create buckets of 5%
    bucket_count = int(100 / bucket_size)
    epoch_buckets = [i * bucket_size for i in range(1, bucket_count + 1)]
    # endregion

    # region create data dict
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
    # endregion

    bar_locations = np.asarray(epoch_buckets)
    bar_width = bucket_size * 0.25

    __create_barchart__(
        data_dict, bar_locations, bar_width, bucket_size,
        suptitle="convergence observations as simulations' progress",
        xlabel="simulations' progress (%)", ylabel=r"c$_{t}$ count",
        figname=figname)
# endregion


# region Pie charts
def piechart_avg_convergence_achieved(figname: str = "GA") -> None:
    # region create data dict
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
    # endregion

    s = len(source_keys)
    grid = (9, 3)
    wedge_labels = ["achieved eq.", "has not achieved eq."]
    fig, axes = plt.subplots(1, s, figsize=grid)
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
        ax.set_xlabel(f"{src_key}", fontproperties=cfg.fp_axis_legend)
    if s == 2:
        plt.legend(labels=wedge_labels, ncol=s, frameon=False, loc="best", bbox_to_anchor=(0.75, -0.20), prop=cfg.fp_axis_legend)
    elif s == 3:
        plt.legend(labels=wedge_labels, ncol=s, frameon=False, loc="best", bbox_to_anchor=(0.6, -0.20), prop=cfg.fp_axis_legend)

    __save_figure__(figname, image_ext)
# endregion


def boxplot_time_to_detect_off_nodes(figname: str = "TSNR") -> None:
    pass


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
    # endregion

    sources_files, source_keys = setup_sources(["SG8-100P", "SG8-1000P", "SG8-2000P"])
    # Q1. Quantas mensagens passam na rede por epoch?
    boxplot_bandwidth(figname="bw_parts")
    # Q2. Existem mais conjuntos de convergencia perto do fim da simulação?
    barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_parts")
    # Q3. Quanto tempo é preciso até observar a primeira convergencia na rede?
    boxplot_first_convergence(figname="fc_parts")
    # Q4. A média dos vectores de distribuição é proxima ao objetivo?
    piechart_avg_convergence_achieved(figname="avgc_pie_parts")
    boxplot_avg_convergence_magnitude_distance(figname="avgc_dist_parts")
    # Q5. Quantas partes são suficientes para um Swarm Guidance  satisfatório?
    boxplot_percent_time_instantaneous_convergence(figname="tic_parts")
    # Q6. Tecnicas de optimização influenciam as questões anteriores?

    sources_files, source_keys = setup_sources(["SG8-1000P", "SG8-Opt"])
    barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_opt")
    boxplot_first_convergence(figname="fc_opt")
    piechart_avg_convergence_achieved(figname="avgc_pie_opt")
    boxplot_avg_convergence_magnitude_distance(figname="avgc_dist_opt")
    boxplot_percent_time_instantaneous_convergence(figname="tic_opt")

    sources_files, source_keys = setup_sources(["SG8-Opt", "SG16-Opt", "SG32-Opt"])
    # Q7. A performance melhora para redes de maior dimensão? (8 vs. 12  vs. 16)
    barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_networks")
    boxplot_first_convergence(figname="fc_networks")
    piechart_avg_convergence_achieved(figname="avgc_pie_networks")
    boxplot_avg_convergence_magnitude_distance(figname="avgc_dist_networks")
    boxplot_percent_time_instantaneous_convergence(figname="tic_networks")

    sources_files, source_keys = setup_sources(["SG8-1000P", "SG8-Opt", "SG16-Opt", "SG32-Opt"])
    # Q8. Qual é o out-degree e in-degree cada rede? Deviam ser usadas constraints?
    boxplot_node_degree(figname="nd-networks")

    sources_files, source_keys = setup_sources(["SG8-1000P", "SG8-ML"])
    barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_msgloss")
    boxplot_first_convergence(figname="fc_msgloss")

    # Q11. Quanto tempo demoramos a detetar falhas de nós com swarm guidance?t_{snr}
    sources_files, source_keys = setup_sources(["SGDBS-T1", "SGDBS-T2", "SGDBS-T3", "HDFS-T1", "HDFS-T2", "HDFS-T3"])
    boxplot_time_to_detect_off_nodes(figname="sgdbs_tsnr")
    # Q12. Os ficheiros sobrevivem mais vezes que no Hadoop Distributed File System?
    # Q13. Se não sobrevivem, quantos epochs sobrevivem com a implementação actual?
    # Q14. Dadas as condições voláteis, qual o impacto na quantidade de convergências instantaneas?
    # Q15. Dadas as condições voláteis, verificamos uma convergência média para \steadystate?
    # Q16. Redes de diferentes tiers, tem resultados significativamente melhores?
