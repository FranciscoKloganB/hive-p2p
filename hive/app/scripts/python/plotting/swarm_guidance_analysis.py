"""
This script collects data
"""
import getopt
import json
import os
import sys
from itertools import zip_longest
from json import JSONDecodeError
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import _matplotlib_configs as cfg

from _matplotlib_configs import color_palette as cp


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


def tokenize(s: str, token: str) -> Tuple[int, int]:
    token_index = s.index("i#o")
    first = int(s[:token_index])
    second = int(s[token_index + len(token):])
    return first, second


def __auto_label__(rects: Any, ax: Any) -> None:
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize="large", fontweight="semibold", color="dimgrey")


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
    plt.savefig(fname, format=figext, bbox_inches="tight")

# endregion


# region Boxplots
def __create_boxplot__(data_dict: Dict[str, Any],
                       suptitle: str, xlabel: str, ylabel: str,
                       figname: str = "", figext: str = "png",
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
        __save_figure__(figname, figext)
    return fig, ax


def __create_double_boxplot__(left_data, right_data,
                              suptitle: str, xlabel: str, ylabel: str,
                              labels: List[str],
                              figname: str = "", figext: str = "png",
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
        __save_figure__(figname, figext)
    return fig, ax


def boxplot_bandwidth(figname: str = "BW") -> None:
    filesize = 47185920  # bytes
    # region create data dict
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                data_dict[src_key].extend(
                    np.asarray(outdata["blocks_moved"]) * outdata["blocks_size"])
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
        left_color=cp[1], right_color=cp[2],
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
                matrices_degrees = outdata["matrices_nodes_degrees"]
                for topology in matrices_degrees:
                    for degrees in topology.values():
                        in_degree, out_degree = tokenize(degrees, "i#o")
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
        left_color=cp[0], right_color=cp[1],
        left_label="in-degree", right_label="out-degree",
        suptitle="Nodes' degrees depending on the cluster's size",
        xlabel="config", ylabel="node degrees",
        labels=source_keys,
        figname=figname, figext=image_ext)


def boxplot_time_to_detect_off_nodes(figname: str = "TSNR") -> None:
    # region create data dict
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)["delay_suspects_detection"].values()
                # worksaround a bug where nodes seem to have taken infinite time to be detected using (0 < x < 15)
                outdata = list(filter(lambda x: x > 0, outdata))
                data_dict[src_key].extend(outdata)

    # endregion
    fig, ax = __create_boxplot__(
        data_dict,
        suptitle="Clusters' time to evict suspect storage nodes",
        xlabel="config", ylabel="epochs",
        savefig=False)

    plt.ylim(0, 10.5)
    plt.axhline(y=5, color=cp[0], linestyle='--', label=r"HDFS t$_{snr}$")
    plt.legend(frameon=False, loc="best", prop=cfg.fp_axis_legend)
    plt.savefig(f"{plots_directory}/{figname}.{image_ext}", format=image_ext, bbox_inches="tight")


def boxplot_terminations(figname: str = "T") -> None:
    # region create data dict
    data_dict = {k: [] for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                data_dict[src_key].append(json.load(outfile)["terminated"])
    # endregion

    __create_boxplot__(
        data_dict,
        suptitle="clusters' termination epochs",
        xlabel="config", ylabel="epoch",
        figname=figname, figext=image_ext)
# endregion


# region Bar charts
def __create_grouped_barchart__(data_dict: Dict[str, Any],
                                bar_locations: np.ndarray, bar_width: float,
                                bucket_size: float,
                                suptitle: str, xlabel: str, ylabel: str,
                                figname: str = "", figext: str = "png",
                                savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()
    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=75, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    plt.xlim(bucket_size - bucket_size * 0.75, 100 + bucket_size * 0.8)
    ax.set_xticks(bar_locations)
    for i in range(len(source_keys)):
        key = source_keys[i]
        epoch_vals = data_dict[key]
        offset = (bar_width * i) - bar_width / len(source_keys)
        ax.bar(bar_locations + offset, epoch_vals, width=bar_width, alpha=0.8)

    ax.legend([str(x) for x in source_keys], frameon=False, loc="best",
              prop=cfg.fp_axis_legend)

    if savefig:
        plt.savefig(f"{plots_directory}/{figname}.{figext}", format=figext,
                    bbox_inches="tight")

    return fig, ax


def __create_barchart__(data_dict: Dict[str, Any],
                        suptitle: str, xlabel: str, ylabel: str,
                        figname: str = "", figext: str = "png",
                        savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()
    plt.suptitle(suptitle, fontproperties=cfg.fp_title)
    plt.xlabel(xlabel, labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    plt.ylabel(ylabel, labelpad=cfg.labels_pad,
               fontproperties=cfg.fp_axis_labels)
    plt.xticks(rotation=75, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    bar_width = 0.66
    bar_locations = np.arange(len(data_dict))
    rects = ax.bar(
        bar_locations, data_dict.values(), bar_width, align="center", alpha=0.8)
    ax.set_xticks(bar_locations)
    ax.set_xticklabels(data_dict.keys())
    __auto_label__(rects, ax)

    if savefig:
        __save_figure__(figname, figext)

    return fig, ax


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

    bar_locations = np.arange(1, len(epoch_buckets) + 1) * bucket_size
    bar_width = bucket_size / (len(data_dict) + 1)

    # bar_width = bucket_size * 0.25
    __create_grouped_barchart__(
        data_dict, bar_locations, bar_width, bucket_size,
        suptitle="convergence observations as simulations' progress",
        xlabel="simulations' progress (%)", ylabel=r"c$_{t}$ count",
        figname=figname, figext=image_ext)


def barchart_successful_simulations(figname: str = "SS") -> None:
    # region create data dict
    data_dict = {k: 0 for k in source_keys}
    for src_key, outfiles_view in sources_files.items():
        count = 0
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                if outdata["terminated"] == epochs:
                    count += 1
        data_dict[src_key] = np.clip(count, 0, epochs)
    # endregion

    __create_barchart__(data_dict,
                        suptitle="Counting successfully terminated simulations",
                        xlabel="config", ylabel=r"number of durable files",
                        figname=figname, figext=image_ext, savefig=False)

    plt.ylim(0, epochs)
    __save_figure__(figname, image_ext)
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
            textprops={'color': 'white', 'weight': 'bold'},
            wedgeprops={'alpha': 0.8}
        )
        ax.set_xlabel(f"{src_key}", fontproperties=cfg.fp_axis_legend)
    if s == 2:
        plt.legend(labels=wedge_labels, ncol=s, frameon=False, loc="best", bbox_to_anchor=(0.75, -0.20), prop=cfg.fp_axis_legend)
    elif s == 3:
        plt.legend(labels=wedge_labels, ncol=s, frameon=False, loc="best", bbox_to_anchor=(0.6, -0.20), prop=cfg.fp_axis_legend)

    __save_figure__(figname, image_ext)
# endregion


if __name__ == "__main__":
    # region args processing
    ns = 8
    epochs = 480
    pde = 0.0
    pml = 0.0
    opt = "off"
    image_ext = "pdf"

    short_opts = "e:o:i:"
    long_opts = ["epochs=", "optimizations=", "image_format="]

    try:
        args, values = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for arg, val in args:
            if arg in ("-e", "--epochs"):
                epochs = int(str(val).strip())
            if arg in ("-o", "--optimizations"):
                opt = str(val).strip()
            if arg in ("-i", "--image_format"):
                image_ext = str(val).strip()

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

    # sources_files, source_keys = setup_sources(["SG8-100P", "SG8-1000P", "SG8-2000P"])
    # # Q1. Quantas mensagens passam na rede por epoch?
    # boxplot_bandwidth(figname="bw_parts")
    # # Q2. Existem mais conjuntos de convergencia perto do fim da simulação?
    # barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_parts")
    # # Q3. Quanto tempo é preciso até observar a primeira convergencia na rede?
    # boxplot_first_convergence(figname="fc_parts")
    # # Q4. A média dos vectores de distribuição é proxima ao objetivo?
    # piechart_avg_convergence_achieved(figname="avgc_pie_parts")
    # boxplot_avg_convergence_magnitude_distance(figname="avgc_dist_parts")
    # # Q5. Quantas partes são suficientes para um Swarm Guidance  satisfatório?
    # boxplot_percent_time_instantaneous_convergence(figname="tic_parts")
    # # Q6. Tecnicas de optimização influenciam as questões anteriores?
    #
    # sources_files, source_keys = setup_sources(["SG8-1000P", "SG8-Opt"])
    # barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_opt")
    # boxplot_first_convergence(figname="fc_opt")
    # piechart_avg_convergence_achieved(figname="avgc_pie_opt")
    # boxplot_avg_convergence_magnitude_distance(figname="avgc_dist_opt")
    # boxplot_percent_time_instantaneous_convergence(figname="tic_opt")
    #
    # sources_files, source_keys = setup_sources(["SG8-Opt", "SG16-Opt", "SG32-Opt"])
    # Q7. A performance melhora para redes de maior dimensão? (8 vs. 12  vs. 16)
    # barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_networks")
    # boxplot_first_convergence(figname="fc_networks")
    # piechart_avg_convergence_achieved(figname="avgc_pie_networks")
    # boxplot_avg_convergence_magnitude_distance(figname="avgc_dist_networks")
    # boxplot_percent_time_instantaneous_convergence(figname="tic_networks")
    #
    # sources_files, source_keys = setup_sources(["SG8-1000P", "SG8-Opt", "SG16-Opt", "SG32-Opt"])
    # # Q8. Qual é o out-degree e in-degree cada rede? Deviam ser usadas constraints?
    # boxplot_node_degree(figname="nd-networks")
    #
    # sources_files, source_keys = setup_sources(["SG8-1000P", "SG8-ML"])
    # barchart_instantaneous_convergence_vs_progress(bucket_size=5, figname="icp_msgloss")
    # boxplot_first_convergence(figname="fc_msgloss")

    # Q11. Quanto tempo demoramos a detetar falhas de nós com swarm guidance?t_{snr}
    sources_files, source_keys = setup_sources(["SGDBS-T1", "SGDBS-T2", "SGDBS-T3"])
    boxplot_time_to_detect_off_nodes(figname="time_to_evict_suspects")
    # Q12. Os ficheiros sobrevivem mais vezes que no Hadoop Distributed File System?
    # Q13. Se não sobrevivem, quantos epochs sobrevivem com a implementação actual?
    sources_files, source_keys = setup_sources(
        ["SGDBS-T1", "SGDBS-T2", "SGDBS-T3", "HDFS-T1", "HDFS-T2", "HDFS-T3"])
    barchart_successful_simulations(figname="successfully_terminated")
    boxplot_terminations(figname="terminations_bp")
    # Q14. Dadas as condições voláteis, qual o impacto na quantidade de convergências instantaneas?
    # Q15. Dadas as condições voláteis, verificamos uma convergência média para \steadystate?
    # Q16. Redes de diferentes tiers, tem resultados significativamente melhores?
