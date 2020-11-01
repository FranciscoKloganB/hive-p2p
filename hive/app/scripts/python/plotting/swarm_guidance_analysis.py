"""
This script collects data
"""
import ast
import sys
import json
import getopt

from itertools import zip_longest
from typing import List, Tuple, Dict


from _matplotlib_configs import *


# region Helpers
def setup_sources(
        source_patterns: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    kdict = {k: [] for k in source_patterns}
    for filename in dirfiles:
        for k in kdict:
            if k in filename:
                kdict[k].append(filename)
                break
    kdict = {k.replace("#", "") if "#" in k else k: v for k, v in kdict.items()}
    klist = list(kdict)
    return kdict, klist


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
# endregion


# region Boxplots
def __create_boxplot__(data_dict: Dict[str, Any],
                       dcolor: Optional[str] = None,
                       dlabel: Optional[str] = None,
                       suptitle: Optional[str] = None,
                       xlabel: Optional[str] = None,
                       ylabel: Optional[str] = None,
                       xtick_rotation: int = 45,
                       showfliers: bool = True,
                       figname: str = "", figext: str = "png",
                       savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()
    switch_tr_spine_visibility(ax)

    bp = plt.boxplot(data_dict.values(), whis=0.75,
                     notch=True, patch_artist=True,
                     showfliers=showfliers, flierprops=outlyer_shape)
    try_coloring(bp, dcolor or color_palette[0], dlabel)

    ax.set_xticklabels(data_dict.keys())

    plt.ylabel(ylabel, labelpad=labels_pad, fontproperties=fp_tick_labels)
    plt.xticks(rotation=xtick_rotation, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    if savefig:
        save_figure(figname, figext, plots_directory)
    return fig, ax


def __create_grouped_boxplot__(datasets: List[List[Any]],
                               dcolors: List[Optional[str]],
                               dlabels: List[Optional[str]],
                               xticks_labels: List[str],
                               xlabel: Optional[str] = None,
                               ylabel: Optional[str] = None,
                               xtick_rotation: int = 45,
                               showfliers: bool = True,
                               figname: str = "", figext: str = "png",
                               savefig: bool = True) -> Tuple[Any, Any]:
    """Creates a figure where each tick has one or more boxplots.

    Args:
        datasets:
            A list containing lists with the boxplot data. For example, if the
            figure is supposed to have one boxplot per tick than, datasets
            argument would look like ``[[a1, b1, c1]]``, if it is supposed to
            have two boxplots per tick than it would be something like
            ``[[a1, b1, c1], [a2, b2, c2]]`` and so on, where ``a1`` is the
            left-most boxplot of the left-most tick and ``cn`` is the right-most
            boxplot of the right-most tick. In this case both examples have
            three ticks, if a ``d`` entry existed, there would four ticks
            instead.
        dcolors:
            The colors used to paint each boxplot or a List of Nones.
        dlabels:
            The description that gives meaning to the colors.
        xticks_labels:
            A description that differentiates each tick from the next.
    """
    fig, ax = plt.subplots()

    switch_tr_spine_visibility(ax)

    colors = 0
    boxplots_per_tick = len(datasets)
    offsets = get_boxplot_offsets(boxplots_per_tick, spacing=0.4)
    for i in range(boxplots_per_tick):
        i_data = datasets[i]
        bp = plt.boxplot(i_data, whis=0.75, widths=0.7,
                         notch=True, patch_artist=True,
                         showfliers=True, flierprops=outlyer_shape,
                         positions=np.array(range(len(i_data))) * boxplots_per_tick + offsets[i])
        colors += try_coloring(bp, dcolors[i], dlabels[i])

    if colors > 0:
        plt.legend(prop=fp_legend, ncol=colors, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.5))

    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=labels_pad, fontproperties=fp_tick_labels)
    if ylabel is not None:
        plt.ylabel(ylabel, labelpad=labels_pad, fontproperties=fp_tick_labels)

    xtick_count = len(xticks_labels)
    xtick_positions = range(0, xtick_count * boxplots_per_tick, boxplots_per_tick)

    plt.xlim(-boxplots_per_tick, xtick_count * boxplots_per_tick)
    plt.xticks(xtick_positions, xticks_labels, rotation=xtick_rotation,
               fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    if savefig:
        save_figure(figname, figext, plots_directory)

    return fig, ax


def boxplot_bandwidth(figname: str) -> None:
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


def boxplot_first_convergence(figname: str, xtick_rotation: int = 45) -> None:
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
        xtick_rotation=xtick_rotation,
        figname=figname, figext=image_ext)


def boxplot_time_in_convergence(figname: str, xtick_rotation: int = 45) -> None:
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
        xtick_rotation=xtick_rotation,
        ylabel=r"sum(c$_{t}$) / termination epoch",
        figname=figname, figext=image_ext)


def boxplot_goal_distances(figname: str, xtick_rotation: int = 45) -> None:
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

    __create_grouped_boxplot__(
        datasets=[psamples, nsamples],
        dcolors=[color_palette[1], color_palette[2]],
        dlabels=["eq. achieved", "eq. not achieved."],
        xticks_labels=srckeys,
        xtick_rotation=xtick_rotation,
        ylabel=r"c$_{dm}$ / cluster size",
        figname=figname, figext=image_ext)


def boxplot_node_degree(figname: str, xtick_rotation: int = 45) -> None:
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

    __create_grouped_boxplot__(
        datasets=[isamples, osamples],
        dcolors=[color_palette[1], color_palette[2]],
        dlabels=["in-degree", "out-degree"],
        xticks_labels=srckeys, xtick_rotation=xtick_rotation,
        ylabel="node degrees",
        figname=figname, figext=image_ext)


def boxplot_time_to_detect_off_nodes(figname: str) -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                outdata = json.load(outfile)["delay_suspects_detection"].values()
                # worksaround a bug where nodes seem to have taken infinite time to be detected using (0 < x < 45)
                # outdata = list(filter(lambda x: 0 < x < 45, outdata))
                outdata = list(filter(lambda x: x > 0, outdata))
                data_dict[src_key].extend(outdata)

    # endregion
    fig, ax = __create_boxplot__(
        data_dict,
        dcolor=color_palette[0],
        dlabel=r"SGDBS t$_{snr}$",
        suptitle="Clusters' time to evict suspect storage nodes",
        ylabel="epochs", showfliers=True,
        savefig=False)

    ax.set_zorder(2)

    plt.axhline(y=5, color=color_palette[-1], alpha=ax_alpha - 0.2,
                linestyle='-', label=r"HDFS Constant t$_{snr}$", zorder=1)

    leg = plt.legend(prop=fp_legend, ncol=2, frameon=False,
                     loc="lower center", bbox_to_anchor=(0.5, -0.5))
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10)

    save_figure(figname, image_ext, plots_directory)


def boxplot_terminations(figname: str) -> None:
    # region create data dict
    data_dict = {k: [] for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                data_dict[src_key].append(json.load(outfile)["terminated"])
    # endregion

    __create_boxplot__(data_dict, ylabel="epoch",
                       figname=figname, figext=image_ext, savefig=False)

    plt.yticks(np.arange(0, 480 + 1, step=80))
    save_figure(figname, image_ext, plots_directory)
# endregion


# region Bar charts
def __create_grouped_barchart__(data_dict: Dict[str, Any],
                                bar_locations: np.ndarray,
                                bar_width: float,
                                bucket_size: float,
                                suptitle: Optional[str] = None,
                                xlabel: Optional[str] = None,
                                ylabel: Optional[str] = None,
                                frameon: bool = False,
                                figname: str = "", figext: str = "png",
                                savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()

    switch_tr_spine_visibility(ax)

    if suptitle is not None:
        plt.suptitle(suptitle, labelpad=title_pad, fontproperties=fp_title)
    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=labels_pad, fontproperties=fp_tick_labels)
    if ylabel is not None:
        plt.ylabel(ylabel, labelpad=labels_pad, fontproperties=fp_tick_labels)

    # xtick_count = len(xticks_labels)
    # xtick_positions = range(0, xtick_count * boxplots_per_tick, boxplots_per_tick)
    #
    # plt.xlim(-boxplots_per_tick, xtick_count * boxplots_per_tick)
    # plt.xticks(xtick_positions, xticks_labels, rotation=xtick_rotation,
    #            fontsize="x-large", fontweight="semibold")

    plt.xticks(rotation=45, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")
    plt.xlim(bucket_size - bucket_size * 0.75, 100 + bucket_size * ax_alpha)

    o = get_barchart_offsets(len(srckeys), bar_width)
    for i in range(len(srckeys)):
        key = srckeys[i]
        epoch_vals = data_dict[key]
        ax.bar(bar_locations + o[i], epoch_vals, width=bar_width, alpha=ax_alpha)

    ax.legend([str(x) for x in srckeys],
              frameon=frameon, prop=fp_legend, ncol=len(srckeys),
              loc="lower center", bbox_to_anchor=(0.5, -0.5))

    if savefig:
        save_figure(figname, figext, plots_directory)

    return fig, ax


def __create_barchart__(data_dict: Dict[str, Any],
                        suptitle: Optional[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        xtick_rotation: int = 45,
                        figname: str = "", figext: str = "png",
                        savefig: bool = True) -> Tuple[Any, Any]:
    fig, ax = plt.subplots()

    switch_tr_spine_visibility(ax)

    if suptitle is not None:
        plt.suptitle(suptitle, labelpad=title_pad, fontproperties=fp_title)
    if xlabel is not None:
        plt.xlabel(xlabel, labelpad=labels_pad, fontproperties=fp_tick_labels)
    if ylabel is not None:
        plt.ylabel(ylabel, labelpad=labels_pad, fontproperties=fp_tick_labels)

    plt.xticks(rotation=xtick_rotation, fontsize="x-large", fontweight="semibold")
    plt.yticks(fontsize="x-large", fontweight="semibold")

    bar_width = 0.66
    bar_locations = np.arange(len(data_dict))
    rects = ax.bar(
        bar_locations, data_dict.values(), bar_width, align="center", alpha=ax_alpha)
    ax.set_xticks(bar_locations)
    ax.set_xticklabels(data_dict.keys())
    __auto_label__(rects, ax)

    if savefig:
        save_figure(figname, figext, plots_directory)

    return fig, ax


def barchart_convergence_vs_progress(figname: str, bucket_size: int = 10) -> None:
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

    __create_grouped_barchart__(
        data_dict, bar_locations, bar_width, bucket_size,
        ylabel=r"c$_{t}$ count", figname=figname, figext=image_ext)


def barchart_successful_simulations(figname: str) -> None:
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
                        ylabel=r"number of durable files",
                        figname=figname, figext=image_ext, savefig=False)

    yticks = np.arange(0, 501, step=100)
    plt.yticks(yticks, fontsize="x-large", fontweight="semibold")
    save_figure(figname, image_ext, plots_directory)
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

    wedge_labels = ["eq. achieved", "eq. not achievied"]

    s = len(srckeys)
    rows = math.ceil(s/4)
    cols = 4
    entry_size = (9 * rows, 3 * cols) if s % 3 == 0 else (12 * rows, 3 * cols)

    fig, axes = plt.subplots(rows, cols, figsize=entry_size)  # figsize=entry_size

    for i, ax in enumerate(axes.flatten()):
        if i >= len(srckeys):
            break
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

    save_figure(figname, image_ext, plots_directory)
# endregion


def barchart_goals_achieved(figname: str, xtick_rotation: int = 45) -> None:
    # region create data dict
    data_dict = {k: 0 for k in srckeys}
    for src_key, outfiles_view in srcfiles.items():
        s = 0
        t = 0
        for filename in outfiles_view:
            filepath = os.path.join(directory, filename)
            with open(filepath) as outfile:
                classifications = json.load(outfile)["topologies_goal_achieved"]
                for result in classifications:
                    t += 1
                    if result is True:
                        s += 1
        data_dict[src_key] = math.floor(s/t * 100) if t > 0 else 0
    # endregion

    __create_barchart__(data_dict,
                        ylabel=r"clusters (%)",
                        xtick_rotation=xtick_rotation,
                        figname=figname, figext=image_ext, savefig=False)

    yticks = np.arange(0, 119, step=20)
    plt.yticks(yticks, fontsize="x-large", fontweight="semibold")
    save_figure(figname, image_ext, plots_directory)


if __name__ == "__main__":
    # region args processing
    epochs = 480
    image_ext = {"pdf", "png"}

    short_opts = "e:i:"
    long_opts = ["epochs=", "image_format="]
    try:
        args, values = getopt.getopt(sys.argv[1:], short_opts, long_opts)

        for arg, val in args:
            if arg in ("-e", "--epochs"):
                epochs = int(str(val).strip())
            if arg in ("-i", "--image_format"):
                image_ext = ast.literal_eval(str(val).strip())

    except ValueError:
        sys.exit("Execution arguments should have the following data types:\n"
                 "  --epochs -e (int)\n"
                 "  --image_format -i (str or set of str), e.g., {'pdf','png'}\n")
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
    # Q2. Existem mais conjuntos de convergencia perto do fim da simulação?
    # Q3. Quanto tempo é preciso até observar a primeira convergencia na rede?
    # Q4. A média dos vectores de distribuição é proxima ao objetivo?
    # Q5. Quantas partes são suficientes para um Swarm Guidance  satisfatório?
    # Q6. Tecnicas de optimização influenciam as questões anteriores?
    # Q7. A performance melhora para redes de maior dimensão? (8 vs. 12  vs. 16)
    # srcfiles, srckeys = setup_sources(["SG8-100P", "SG8-1000P", "SG8-2000P"])
    # boxplot_bandwidth(figname="Bandwidth-Consumption")
    # barchart_convergence_vs_progress(figname="Convergence-Progress_BC_Parts")
    # boxplot_first_convergence(figname="First-Convergence_BP_Parts")
    # boxplot_time_in_convergence(figname="Time-in-Convergence_BP_Parts")
    # boxplot_goal_distances(figname="Goal-Distance_BP_Parts")
    # piechart_goals_achieved(figname="Goals-Achieved_PC_Parts")

    # srcfiles, srckeys = setup_sources(["SG8-ML", "SG8#", "SG16#", "SG32#"])
    # barchart_convergence_vs_progress(figname="Convergence-Progress_BC_Sizes")
    # boxplot_first_convergence(figname="First-Convergence_BP-Sizes")
    # boxplot_time_in_convergence(figname="Time-in-Convergence_BP_Sizes")
    # boxplot_goal_distances(figname="Goal-Distance_BP_Sizes")
    # piechart_goals_achieved(figname="Goals-Achieved_PC_Sizes")

    # srcfiles, srckeys = setup_sources(["SG8-Opt", "SG16-Opt", "SG32-Opt"])
    # barchart_convergence_vs_progress(figname="Convergence-Progress_BC_Sizes-Opt")
    # boxplot_first_convergence(figname="First-Convergence_BP-Sizes-Opt")
    # boxplot_time_in_convergence(figname="Time-in-Convergence_BP_Sizes-Opt")
    # boxplot_goal_distances(figname="Goal-Distance_BP_Sizes-Opt")
    # piechart_goals_achieved(figname="Goals-Achieved_PC_Sizes-Opt")

    # Q11. Qual é o out-degree e in-degree cada rede? Deviam ser usadas constraints?
    # srcfiles, srckeys = setup_sources(["SGDBS-T1", "SG8#", "SG16#", "SG32#"])
    # boxplot_node_degree(figname="Node-Degrees_BP_SG")

    # Q12. Quanto tempo demoramos a detetar falhas de nós com swarm guidance? t_{snr}
    # srcfiles, srckeys = setup_sources(["SGDBS-T1", "SGDBS-T2", "SGDBS-T3"])
    # boxplot_time_to_detect_off_nodes(figname="Time-to-Evict-Suspects_BP_SGDBS")

    # Q13. Os ficheiros sobrevivem mais vezes que no Hadoop Distributed File System?
    # Q14. Se não sobrevivem, quantos epochs sobrevivem com a implementação actual?
    # Q15. Redes de diferentes tiers, tem resultados significativamente melhores?
    srcfiles, srckeys = setup_sources(
        ["SGDBS-T1", "SGDBS-T2", "SGDBS-T3", "HDFS-T1", "HDFS-T2", "HDFS-T3"])
    barchart_successful_simulations(figname="Successful-Simulations_SGDBS-HDFS")
    # boxplot_terminations(figname="Terminations_BP_SGDBS-HDFS")

    # Q16. Dadas as condições voláteis, qual o impacto na quantidade de convergências instantaneas?
    # Q17. Dadas as condições voláteis, verificamos uma convergência média para \steadystate?
    # srcfiles, srckeys = setup_sources(["SGDBS-T1", "SGDBS-T2", "SGDBS-T3"])
    # piechart_goals_achieved(figname="SGDBS-Avg-Convergence_PC")
    # boxplot_goal_distances(figname="SGDBS-Goal-Distance_BP")
    # boxplot_time_in_convergence(figname="SGDBS-Time-Convergence_BP")

    # region ---- IMPROVED READABILITY PLOTS SECTION BELOW ----
    # srcfiles, srckeys = setup_sources([
    #     "SG8-100P", "SG8-1000P", "SG8-2000P",
    #     "SG8-ML", "SG8#", "SG16#", "SG32#",
    #     "SG8-Opt", "SG16-Opt", "SG32-Opt"
    # ])
    # boxplot_first_convergence("First-Convergence_BP", xtick_rotation=90)
    # boxplot_time_in_convergence("Time-in-Convergence_BP", xtick_rotation=90)
    # boxplot_goal_distances("Goal-Distance_BP", xtick_rotation=90)
    # barchart_goals_achieved("Goals-Achieved_BC", xtick_rotation=90)
    # endregion

    # srcfiles, srckeys = setup_sources(["SGDBS", "SG8#", "SG16#", "SG32#"])
    # boxplot_node_degree("Nodes-Degrees_BP_SGDBS")

    # srcfiles, srckeys = setup_sources(["SGDBS-T1", "SGDBS-T2", "SGDBS-T3"])
    # barchart_goals_achieved("Goals-Achieved_BC_SGDBS")
    # boxplot_goal_distances("Goal-Distance_BP_SGDBS")
