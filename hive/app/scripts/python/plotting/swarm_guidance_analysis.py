"""
This script collects data
"""
import os
import sys
import json
import math
import getopt

from itertools import zip_longest
from json import JSONDecodeError
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from _matplotlib_configs import *


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
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(ax_alpha)
    plt.setp(bp['whiskers'], color="#000000")
    plt.setp(bp['caps'], color="#000000")
    plt.setp(bp['medians'], color="#000000")


def __prop_legend__(color: str, label: str, lw: int = 10) -> None:
    plt.plot([], c=color, label=label, markersize=5, linewidth=lw)


def __save_figure__(figname: str, figext: str = "png") -> None:
    fname = f"{plots_directory}/{figname}.{figext}"
    plt.savefig(fname, format=figext, bbox_inches="tight")
    plt.close('all')
# endregion


# region Boxplots
def __create_boxplot__(data_dict: Dict[str, Any],
                       suptitle: str, xlabel: str, ylabel: str,
                       showfliers: bool = True,
                       figname: str = "", figext: str = "png",
                       savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    bp = ax.boxplot(data_dict.values(), showfliers=showfliers,
               flierprops=outlyer_shape, whis=0.75, notch=True)
    ax.set_xticklabels(data_dict.keys())
    # plt.suptitle(suptitle, fontproperties=fp_title)
    # plt.xlabel(xlabel, labelpad=labels_pad, fontproperties=fp_axis_labels)
    plt.ylabel(ylabel, labelpad=labels_pad, fontproperties=fp_tick_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    plt.setp(bp['medians'], color="#000000")
    
    if savefig:
        __save_figure__(figname, figext)
    return fig, ax


def __create_double_boxplot__(left_data, right_data,
                              suptitle: str, xlabel: str, ylabel: str,
                              labels: List[str],
                              figname: str = "", figext: str = "png",
                              lcolor: Optional[str] = None,
                              rcolor: Optional[str] = None,
                              llabel: Optional[str] = None,
                              rlabel: Optional[str] = None,
                              savefig: bool = True) -> Tuple[Any, Any]:

    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    bpl = plt.boxplot(left_data, showfliers=False, notch=True,
                      whis=0.75, widths=0.7, patch_artist=True,
                      positions=np.array(range(len(left_data))) * 2.0 - 0.4)
    bpr = plt.boxplot(right_data, showfliers=False, notch=True,
                      whis=0.75, widths=0.7, patch_artist=True,
                      positions=np.array(range(len(right_data))) * 2.0 + 0.4)

    cols = 0
    if lcolor:
        cols += 1
        __set_box_color__(bpl, lcolor)
        if llabel:
            __prop_legend__(lcolor, llabel)

    if rcolor:
        cols += 1
        __set_box_color__(bpr, rcolor)
        if rlabel:
            __prop_legend__(rcolor, rlabel)

    if llabel or rlabel:
        plt.legend(prop=fp_legend, ncol=cols, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.5))

    # plt.suptitle(suptitle, fontproperties=fp_title)
    # plt.xlabel(xlabel, labelpad=labels_pad, fontproperties=fp_axis_labels)
    plt.ylabel(ylabel, labelpad=labels_pad, fontproperties=fp_tick_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    label_count = len(labels)
    plt.xticks(range(0, label_count * 2, 2), labels, rotation=45, fontsize="x-large", fontweight="semibold")
    plt.xlim(-2, label_count * 2)

    if savefig:
        __save_figure__(figname, figext)
    return fig, ax


def boxplot_bandwidth(figname: str = "BW") -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)
                blocks_size = outdata["blocks_size"] / 1024 / 1024  # B to MB
                data_dict[src_key].extend(
                    np.asarray(outdata["blocks_moved"]) * blocks_size)
    # endregion

    __create_boxplot__(
        data_dict,
        suptitle="clusters' bandwidth expenditure per epoch",
        xlabel="config", ylabel="moved blocks (MB)",
        showfliers=False,
        figname=figname, figext=image_ext)


def boxplot_first_convergence(figname: str = "FIC") -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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
        xlabel="config", ylabel="epoch", showfliers=False,
        figname=figname, figext=image_ext)


def boxplot_time_in_convergence(figname: str = "TIC") -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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
        suptitle="clusters' time spent in convergence", showfliers=False,
        xlabel="config", ylabel=r"sum(c$_{t}$) / termination epoch",
        figname=figname, figext=image_ext)


def boxplot_goal_distances(figname: str = "MD") -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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
    for src_key in srckeys:
        psamples.append(data_dict[src_key][0])
        nsamples.append(data_dict[src_key][1])

    __create_double_boxplot__(
        psamples, nsamples,
        lcolor=color_palette[1], rcolor=color_palette[2],
        llabel="achieved eq.", rlabel="has not achieved eq.",
        suptitle="clusters' distance to the select equilibrium",
        xlabel="config", ylabel=r"c$_{dm}$ / cluster size",
        labels=srckeys,
        figname=figname, figext=image_ext)


def boxplot_node_degree(figname: str = "ND") -> None:
    """The integral part of the float value is the
    in-degree, the decimal part is the out-degree."""
    # region create data dict
    data_dict = {k: [[], []] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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
    for src_key in srckeys:
        isamples.append(data_dict[src_key][0])
        osamples.append(data_dict[src_key][1])

    __create_double_boxplot__(
        isamples, osamples,
        lcolor=color_palette[0], rcolor=color_palette[1],
        llabel="in-degree", rlabel="out-degree",
        suptitle="Nodes' degrees depending on the cluster's size",
        xlabel="config", ylabel="node degrees",
        labels=srckeys,
        figname=figname, figext=image_ext)


def boxplot_time_to_detect_off_nodes(figname: str = "TSNR") -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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
        xlabel="config", ylabel="epochs", showfliers=False,
        savefig=False)

    ax.set_zorder(2)

    plt.axhline(y=5, color=color_palette[-1], alpha=ax_alpha - 0.2,
                linestyle='--', label=r"HDFS t$_{snr}$", zorder=1)

    leg = plt.legend(prop=fp_legend, ncol=1, frameon=False,
                     loc="lower center", bbox_to_anchor=(0.5, -0.5))

    for legobj in leg.legendHandles:
        legobj.set_markersize(12)

    __save_figure__(figname, image_ext)


def boxplot_terminations(figname: str = "T") -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                data_dict[src_key].append(json.load(outfile)["terminated"])
    # endregion

    __create_boxplot__(
        data_dict,
        suptitle="clusters' termination epochs",
        xlabel="config", ylabel="epoch",
        figname=figname, figext=image_ext, savefig=False)

    plt.yticks(np.arange(0, epochs + 1, step=80))
    __save_figure__(figname, image_ext)
# endregion


# region Bar charts
def __get_offsets__(nbars: int, bwidth: float = 1) -> np.ndarray:
    lim = math.floor(nbars / 2)
    sublocations = np.arange(-lim, lim + 1)
    if nbars % 2 == 0:
        sublocations = list(filter(lambda x: x != 0, sublocations))
        sublocations = list(map(
            lambda x: x + (0.5 if x < 0 else -0.5), sublocations))
        return np.asarray(sublocations) * bwidth
    return sublocations * bwidth


def __create_grouped_barchart__(data_dict: Dict[str, Any],
                                bar_locations: np.ndarray, bar_width: float,
                                bucket_size: float,
                                suptitle: str, xlabel: str, ylabel: str,
                                frameon: bool = False,
                                figname: str = "", figext: str = "png",
                                savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.suptitle(suptitle, fontproperties=fp_title)
    plt.xlabel(xlabel, labelpad=labels_pad,
               fontproperties=fp_tick_labels)
    plt.ylabel(ylabel, labelpad=labels_pad,
               fontproperties=fp_tick_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    plt.xlim(bucket_size - bucket_size * 0.75, 100 + bucket_size * ax_alpha)
    # ax.set_xticks(bar_locations)

    o = __get_offsets__(len(srckeys), bar_width)
    for i in range(len(srckeys)):
        key = srckeys[i]
        epoch_vals = data_dict[key]
        ax.bar(bar_locations + o[i], epoch_vals, width=bar_width, alpha=ax_alpha)

    ax.legend([str(x) for x in srckeys],
              frameon=frameon, prop=fp_legend, ncol=len(srckeys),
              loc="lower center", bbox_to_anchor=(0.5, -0.5))

    if savefig:
        __save_figure__(figname, figext)

    return fig, ax


def __create_barchart__(data_dict: Dict[str, Any],
                        suptitle: str, xlabel: str, ylabel: str,
                        figname: str = "", figext: str = "png",
                        savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # plt.suptitle(suptitle, fontproperties=fp_title)
    # plt.xlabel(xlabel, labelpad=labels_pad, fontproperties=fp_axis_labels)
    plt.ylabel(ylabel, labelpad=labels_pad,
               fontproperties=fp_tick_labels)
    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    bar_width = 0.66
    bar_locations = np.arange(len(data_dict))
    rects = ax.bar(
        bar_locations, data_dict.values(), bar_width, align="center", alpha=ax_alpha)
    ax.set_xticks(bar_locations)
    ax.set_xticklabels(data_dict.keys())
    __auto_label__(rects, ax)

    if savefig:
        __save_figure__(figname, figext)

    return fig, ax


def barchart_convergence_vs_progress(
        bucket_size: int = 10, figname: str = "ICC") -> None:
    # region create buckets of 5%
    bucket_count = int(100 / bucket_size)
    epoch_buckets = [i * bucket_size for i in range(1, bucket_count + 1)]
    # endregion

    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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
        suptitle="convergence observations as simulations progress",
        xlabel="simulations progress (%)", ylabel=r"c$_{t}$ count",
        figname=figname, figext=image_ext)


def barchart_successful_simulations(figname: str = "SS") -> None:
    # region create data dict
    data_dict = {k: 0 for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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

    yticks = np.arange(0, epochs + 1, step=80)
    plt.yticks(yticks, fontsize="x-large", fontweight="semibold")
    __save_figure__(figname, image_ext)
# endregion


# region Pie charts
def piechart_goals_achieved(figname: str = "GA") -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
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

    s = len(srckeys)
    grid = (9, 3) if s % 3 == 0 else (12, 3)
    wedge_labels = ["achieved eq.", "has not achieved eq."]
    fig, axes = plt.subplots(1, s, figsize=grid)
    # plt.suptitle("clusters (%) achieving the selected equilibrium on average", fontproperties=fp_title, x=0.51)
    for i, ax in enumerate(axes.flatten()):
        ax.axis('equal')
        src_key = srckeys[i]
        wedges, _, _ = ax.pie(
            data_dict[src_key],
            explode=(0.025, 0.025),
            startangle=90, autopct='%1.1f%%',
            labels=wedge_labels, labeldistance=None,
            textprops={'color': 'white', 'weight': 'bold'},
            wedgeprops={'alpha': ax_alpha}
        )
        ax.set_xlabel(f"{src_key}", fontproperties=fp_pctick_labels)

    offset = (0.6, -0.5) if s % 3 == 0 else (0.05, -0.5)

    plt.legend(labels=wedge_labels, prop=fp_legend, ncol=s,
               frameon=False, loc="lower right", bbox_to_anchor=offset)

    __save_figure__(figname, image_ext)
# endregion


if __name__ == "__main__":
    # region args processing
    epochs = 480
    image_ext = "pdf"

    short_opts = "e:i:"
    long_opts = ["epochs=", "image_format="]
    try:
        args, values = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for arg, val in args:
            if arg in ("-e", "--epochs"):
                epochs = int(str(val).strip())
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
    dirfiles = list(filter(lambda f: f.endswith(".json"), os.listdir(directory)))

    # Q1. Quantas mensagens passam na rede por epoch?
    srcfiles, srckeys = setup_sources(["SG8-100P", "SG8-1000P", "SG8-2000P"])
    boxplot_bandwidth(figname="bw_parts")

    # Q2. Existem mais conjuntos de convergencia perto do fim da simulação?
    # Q3. Quanto tempo é preciso até observar a primeira convergencia na rede?
    # Q4. A média dos vectores de distribuição é proxima ao objetivo?
    # Q5. Quantas partes são suficientes para um Swarm Guidance  satisfatório?
    # Q6. Tecnicas de optimização influenciam as questões anteriores?
    # Q7. A performance melhora para redes de maior dimensão? (8 vs. 12  vs. 16)
    srcfiles, srckeys = setup_sources(["SG8-100P", "SG8-1000P", "SG8-2000P"])
    barchart_convergence_vs_progress(figname="Convergence-Progress_BC_Parts")
    boxplot_first_convergence(figname="First-Convergence_BP_Parts")
    boxplot_time_in_convergence(figname="Time-in-Convergence_BP_Parts")
    boxplot_goal_distances(figname="Goal-Distance_BP_Parts")
    piechart_goals_achieved(figname="Goals-Achieved_PC_Parts")

    srcfiles, srckeys = setup_sources(["SG8-ML", "SG8", "SG16", "SG32"])
    barchart_convergence_vs_progress(figname="Convergence-Progress_BC_Sizes")
    boxplot_first_convergence(figname="First-Convergence_BP-Sizes")
    boxplot_time_in_convergence(figname="Time-in-Convergence_BP_Sizes")
    boxplot_goal_distances(figname="Goal-Distance_BP_Sizes")
    piechart_goals_achieved(figname="Goals-Achieved_PC_Sizes")

    srcfiles, srckeys = setup_sources(["SG8-Opt", "SG16-Opt", "SG32-Opt"])
    barchart_convergence_vs_progress(figname="Convergence-Progress_BC_Sizes-Opt")
    boxplot_first_convergence(figname="First-Convergence_BP-Sizes-Opt")
    boxplot_time_in_convergence(figname="Time-in-Convergence_BP_Sizes-Opt")
    boxplot_goal_distances(figname="Goal-Distance_BP_Sizes-Opt")
    piechart_goals_achieved(figname="Goals-Achieved_PC_Sizes-Opt")

    # Q11. Qual é o out-degree e in-degree cada rede? Deviam ser usadas constraints?
    srcfiles, srckeys = setup_sources(["SGDBS-T1", "SG8", "SG16", "SG32"])
    boxplot_node_degree(figname="Node-Degrees_BP_SG")

    srcfiles, srckeys = setup_sources(["SGDBS-T1", "SGDBS-T2", "SGDBS-T3"])
    # Q12. Quanto tempo demoramos a detetar falhas de nós com swarm guidance? t_{snr}
    boxplot_time_to_detect_off_nodes(figname="Time-to-Evict-Suspects_BP_SGDBS")
    # Q13. Os ficheiros sobrevivem mais vezes que no Hadoop Distributed File System?
    # Q14. Se não sobrevivem, quantos epochs sobrevivem com a implementação actual?
    # Q15. Redes de diferentes tiers, tem resultados significativamente melhores?
    srcfiles, srckeys = setup_sources(
        ["SGDBS-T1", "SGDBS-T2", "SGDBS-T3", "HDFS-T1", "HDFS-T2", "HDFS-T3"])
    barchart_successful_simulations(figname="Successful-Simulations_SBDBS-HDFS")
    boxplot_terminations(figname="Terminations_BP_SGDBS-HDFS")
    # Q16. Dadas as condições voláteis, qual o impacto na quantidade de convergências instantaneas?
    # Q17. Dadas as condições voláteis, verificamos uma convergência média para \steadystate?
    srcfiles, srckeys = setup_sources(["SGDBS-T1", "SGDBS-T2", "SGDBS-T3"])
    piechart_goals_achieved(figname="SGDBS-Avg-Convergence_PC")
    boxplot_goal_distances(figname="SGDBS-Goal-Distance_BP")
    boxplot_time_in_convergence(figname="SGDBS-Time-Convergence_BP")
