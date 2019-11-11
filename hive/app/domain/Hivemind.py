import os
import json
import random
import numpy as np
import pandas as pd
import domain.metropolis_hastings as mh
import logging as log

from pathlib import Path
from domain.Worker import Worker
from utils.convertions import str_copy
from domain.Enums import Status, HttpCodes
from domain.helpers.FileData import FileData
from utils.randoms import excluding_randrange
from typing import List, Dict, Tuple, Optional
from domain.SharedFilePart import SharedFilePart
from domain.helpers.ConvergenceData import ConvergenceData
from globals.globals import SHARED_ROOT, SIMULATION_ROOT, READ_SIZE, DEFAULT_COLUMN


class Hivemind:
    # region docstrings
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar workers: keeps track workers objects in the simulation
    :type dict<str, domain.Worker>
    :ivar workers_uptime: contains each worker node uptime, used to calculate kill probability
    :type dict<str, float>
    :ivar worker_status: keeps track workers objects in the simulation regarding their health
    :type dict<domain.Worker, domain.Status(Enum)>
    :ivar shared_files: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar sf_data: part_name is a key to containing general information about the file
    :type tuple<str, domain.Helpers.FileData>
    :ivar max_stages: number of stages the hive has to converge to the ddv before simulation is considered failed
    :type int
    """
    # endregion

    # region class variables, instance variables and constructors
    TRUE_FALSE = [True, False]

    def __init__(self, simfile_name):
        """
        :param simfile_name: path to json file containing the parameters this simulation should execute with
        :type str
        """
        simfile_path = os.path.join(SIMULATION_ROOT, simfile_name)
        with open(simfile_path) as input_file:
            json_obj = json.load(input_file)
            # Init basic simulation variables
            self.shared_files = {}
            self.sf_data = {}
            self.workers = {}
            self.worker_status = {}
            self.workers_uptime = json_obj['nodes_uptime']
            self.max_stages = json_obj['max_stages']
            # Create the P2P network nodes (domain.Workers) without any job
            self.__init_workers([*self.workers_uptime.keys()])
            # Read and split all shareable files specified on the input
            self.__split_all_shared_files([*json_obj['shared'].keys()])
            # For all shareable files, set that shared file_routing table
            self.__synthesize_shared_files_transition_matrices(json_obj['shared'])
            # Distribute files before starting simulation
            self.__uniformely_assign_parts_to_workers(self.shared_files, enforce_online=False)
    # endregion

    # region domain.Worker related methods
    def __init_workers(self, worker_names):
        """
        Creates worker objects that knows this Hivemind and starts tracking their health
        :param worker_names:
        :type list<str>
        """
        for name in worker_names:
            worker = Worker(self, name)
            self.workers[name] = worker
            self.worker_status[worker] = Status.ONLINE

    def __uniformely_assign_parts_to_workers(self, shared_files_dict, enforce_online=True):
        """
        Distributes received file parts over the Hive network.
        :param shared_files_dict: receives anyone's dictionary of <file_name, dict<file_number, SharedFilePart>>
        :type dict<str, dict<int, domain.SharedFilePart>>
        :param enforce_online: makes sure receiving workers are online.
        :type bool
        """
        workers_objs = self.__filter_and_map_online_workers() if enforce_online else [*self.worker_status.keys()]
        for file_name, part_number in shared_files_dict.items():
            # Retrive state labels from the file distribution vector
            worker_names = [*self.sf_data[file_name].desired_distribution.index]
            # Quickly filter from the workers which ones are online, fast version of set(a).intersection(b),
            # Do not change positions of file_sharers_names with worker_objs...
            # Doing so changes would make us obtain list<str> containing their names instead of list<domain.Workers>
            choices = [*filter(set(worker_names).__contains__, workers_objs)]
            for part in part_number.values():
                # Randomly choose a destinatary worker from possible choices and give him the shared file part
                np.random.choice(choices).receive_part(part, no_check=True)
    # endregion

    # region file partitioning methods
    def __split_all_shared_files(self, file_names):
        """
        Obtains the path of all files that are going to be divided for sharing simulation and splits them into parts
        :param file_names: a list containing the name of the files to be shared with their extensions included
        :type str
        """
        for file_name in file_names:
            self.__split_shared_file(os.path.join(SHARED_ROOT, file_name))

    def __split_shared_file(self, file_path):
        """
        Reads contents of the file in 2KB blocks and encapsulates them along with their ID and SHA256
        :param file_path: path to an arbitrary file to persist on the hive network
        :type str
        """
        # file_extension = Path(file_path).resolve().suffix
        sf_name = Path(file_path).resolve().stem
        with open(file_path, "rb") as shared_file:
            file_parts = {}
            part_number = 0
            while True:
                read_buffer = shared_file.read(READ_SIZE)
                if read_buffer:
                    part_number = part_number + 1
                    shared_file_part = SharedFilePart(
                        sf_name,
                        part_number,
                        read_buffer
                    )
                    file_parts[part_number] = shared_file_part
                else:
                    # Keeps track of how many parts the file was divided into
                    self.shared_files[sf_name] = file_parts
                    break
            self.sf_data[sf_name] = FileData(file_name=sf_name, parts_count=part_number)
    # endregion

    # region metropolis hastings and transition vector assignment methods
    def __synthesize_transition_matrix(self, adj_matrix, desired_distribution, states):
        """
        :param states: list of worker names who form an hive
        :type list<str>
        :param adj_matrix: adjacency matrix representing connections between various states
        :type list<list<float>>
        :param desired_distribution: column vector representing the distribution that must be achieved by the workers
        :returns: A matrix with named lines and columns with the computed transition matrix
        :rtype pandas.DataFrame
        """
        transition_matrix = mh.metropolis_algorithm(adj_matrix, desired_distribution, column_major_out=True)
        return pd.DataFrame(transition_matrix, index=states, columns=states)

    def __synthesize_shared_files_transition_matrices(self, shared_dict):
        """
        For all keys in the dictionary, obtain file names, the respective adjacency matrix and the desired distribution
        then calculate the transition matrix using metropolis hastings algorithm and feed the result to each worker who
        is a contributor for the survivability of that file
        :param shared_dict: maps file name with extensions to a dictinonary with three keys containing worker_labels who
        are going to receive the file parts associated wih the named file, along with the transition vectors before
        being metropolis hastings processed as well as the desired distributions
        """
        for ext_file_name, markov_chain_data in shared_dict.items():
            sf_name = Path(ext_file_name).resolve().stem
            labels = markov_chain_data['state_labels']
            adj_matrix = markov_chain_data['adj_matrix']
            desired_distribution = markov_chain_data['ddv']
            # Setting the trackers in this phase speeds up simulation
            self.__init_file_data(sf_name, adj_matrix, desired_distribution, labels)
            # Compute transition matrix
            transition_matrix = self.__synthesize_transition_matrix(adj_matrix, desired_distribution, labels)
            # Split transition matrix into column vectors
            self.__set_routing_tables(sf_name, labels, transition_matrix)
    # endregion

    # region simulation execution methods
    def execute_simulation(self):
        """
        Runs a stochastic swarm guidance algorithm applied to a P2P network
        """
        online_workers_list = self.__filter_and_map_online_workers()
        for stage in range(self.max_stages):
            online_workers_list = self.__remove_some_workers(online_workers_list, stage)
            for worker in online_workers_list:
                worker.route_parts()
            self.__process_stage_results(stage)

    def __care_taking(self, stage: int, sf_data: FileData, dead_worker: Worker) -> None:
        """
        :param int stage: number representing the discrete time step the simulation is currently at
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        :param Worker dead_worker: instance object corresponding to the worker who left the network
        """
        replacement_dict: Dict[str, str] = self.__heal_hive(sf_data, dead_worker)
        if replacement_dict:
            sf_data.fwrite("Node replacement at stage {}, replaced: {}\n".format(stage, replacement_dict))
        else:
            self.__contract_hive(sf_data, dead_worker.name)  # contraction only occurs if healing fails
            sf_data.fwrite("Hive contraction at stage {}, due to worker: ".format(stage, dead_worker.name))

    def __init_recovery_protocol(self, sf_data: FileData, mock: Dict[int, SharedFilePart] = None) -> None:
        """
        Starts file recovering by asking the best in the Hive still alive to run a file reconstruction algorithm
        :param FileData sf_data: instance object containing information about the file to be recovered
        :param Dict[int, SharedFilePart] mock: recovery will be accomplished by passing files from dead to living worker
        """
        # TODO future-iterations:
        #  get the next best ONLINE node, for now assume best node is always online, which is likely to be true anyway!
        best_worker_name = sf_data.highest_density_node_label
        worker = self.workers[best_worker_name]
        if mock:
            sf_data.fwrite("Asking best worker, {}, to initialize recovery protocol mock...".format(best_worker_name))
            worker.receive_parts(mock, sf_data.file_name, no_check=True)
        else:
            sf_data.fwrite("Asking best worker, {}, to initialize recovery protocols...".format(best_worker_name))
            worker.init_recovery_protocol(sf_data.file_name)

    def receive_complaint(self, suspects_name: str, sf_name: str = None):
        # TODO future-iterations:
        #  1. register complaint
        #  2. when byzantine complaints > threshold
        #    2.1. find away of obtaining shared_file_names user had
        #    2.2. discover the files the node used to share, probably requires yet another sf_strucutre
        #    2.3. ask the next highest density node that is alive to rebuild dead nodes' files
        if sf_name and suspects_name:
            pass  # maybe sf_name can help when enough complaints are received, reanalyze this param at a later date.
        log.warning("receive_complaint is only a mock. method needs to be implemented...")

    def route_file_part(self, dest_worker_name: str, part: SharedFilePart):
        """
        :param dest_worker_name: destinatary of the file part
        :type str
        :param part: the file part to send to specified worker
        :type domain.SharedFilePart
        :returns: http codes based status of destination worker
        :rtype int
        """
        dest_status = self.worker_status[dest_worker_name]
        if dest_status == Status.ONLINE:
            self.workers[dest_worker_name].receive_part(part, no_check=True)
            return HttpCodes.OK
        elif dest_status == Status.OFFLINE:
            return HttpCodes.SERVER_DOWN
        elif dest_status == Status.SUSPECT:
            return HttpCodes.TIME_OUT
        else:
            return HttpCodes.NOT_FOUND

    def redistribute_file_parts(self, parts):
        """
        :param parts: The file parts to be redistributed in the system
        :type dict<str, domain.SharedFilePart>
        """
        if parts:
            self.__uniformely_assign_parts_to_workers(parts, enforce_online=True)
    # endregion

    # region helper methods
    # region setup
    def __init_file_data(
            self, sf_name: str, adj_matrix: List[List[int]], desired_distribution: List[float], labels: List[str]):
        """
        :param sf_name: the name of the file to be tracked by the hivemind
        :type str
        :param adj_matrix: matrix with connections between nodes, 1 if possible to go from node i to j, else 0
        :type list<list<int>>
        :param desired_distribution: the desired distribution vector of the given named file
        :type list<float>
        :param labels: name of the workers belonging to the hive, i.e.: keepers or sharers of the files
        :type list<str>
        """
        sf_data = self.sf_data[sf_name]
        sf_data.adjacency_matrix = pd.DataFrame(adj_matrix, index=labels, columns=labels)
        sf_data.desired_distribution = pd.DataFrame(desired_distribution, index=labels)
        sf_data.current_distribution = pd.DataFrame([0] * len(desired_distribution), index=labels)
        sf_data.convergence_data = ConvergenceData()
    # endregion

    # region stage processing
    def __remove_some_workers(self, online_workers: List[Worker], stage: int = None) -> List[Worker]:
        """
        For each online worker, if they are online, see if they remain alive for the next stage or if they die,
        according to their uptime record.
        :param online_workers: collection of workers that are known to be online
        :type list<domain.Worker>
        :returns surviving_workers: subset of online_workers, who weren't selected to be removed from the hive
        :rtype list<domain.Worker>
        """
        surviving_workers: List[Worker] = []
        for worker in online_workers:
            uptime: float = self.workers_uptime[worker.name] / 100
            remains_alive: bool = np.random.choice(Hivemind.TRUE_FALSE, p=[uptime, 1 - uptime])
            if remains_alive:
                surviving_workers.append(worker)
            else:
                self.__remove_worker(worker, stage=stage)
        return surviving_workers

    def __remove_worker(self, dead_worker: Worker, stage: int = None) -> None:
        """
        Changes a Worker's status to Status.OFFLINE, it then register a file persistency failure or recovers accordingly
        :param Worker dead_worker: Worker instance to be removed to be forcefully disconnected
        :param int stage: number representing the discrete time step the simulation is currently at
        """
        sf_parts: Dict[str, Dict[int, SharedFilePart]] = dead_worker.get_all_parts()
        sf_parts_dict = sf_parts.items()

        self.worker_status[dead_worker.name] = Status.OFFLINE

        sf_to_drop: List[str] = []
        for sf_name, sf_id_sfp_dict in sf_parts_dict:
            sf_data: FileData = self.sf_data[sf_name]
            threshold: int = sf_data.get_failure_threshold()
            if len(sf_id_sfp_dict) > threshold:
                self.__workers_stop_tracking_shared_file(sf_data)
                sf_to_drop.append(sf_name)
                sf_data.fwrite("Failure stage {}. {} had more than {} parts\n".format(stage, dead_worker.name, threshold))
            else:
                self.__care_taking(stage, sf_data, dead_worker)
                self.__init_recovery_protocol(sf_data, mock=sf_parts[sf_name])
        self.__hivemind_stops_tracking_shared_files(sf_to_drop)

    def __process_stage_results(self, stage: int) -> None:
        """
        Obtains all workers' densities regarding each shared file and logs progress in the system accordingly
        :param int stage: number representing the discrete time step the simulation is currently at
        """
        if stage == self.max_stages - 1:
            for sf_data in self.sf_data.values():
                sf_data.fwrite("Reached final stage... Executing tear down processes. Summary below:\n")
                sf_data.convergence_data.save_sets_and_reset()
                sf_data.fwrite(str(sf_data.convergence_data))
                sf_data.fclose()
            exit(0)
        else:
            print("Stage {}".format(stage))
            for sf_data in self.sf_data.values():
                # retrieve from each worker their part counts for current sf_name and update convergence data
                self.__request_file_counts(sf_data)
                # when all queries for a file are done, verify convergence for data.file_name
                self.__check_file_convergence(stage, sf_data)

    def __request_file_counts(self, sf_data: FileData) -> None:
        """
        Updates inputed FileData.current_distribution with current density values for each worker sharing the file
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        """
        worker_names = [*sf_data.desired_distribution.index]
        for name in worker_names:
            if self.worker_status[name] != Status.ONLINE:
                sf_data.current_distribution.at[name, DEFAULT_COLUMN] = 0
            else:
                worker = self.workers[name]  # get worker instance corresponding to name
                sf_data.current_distribution.at[name, DEFAULT_COLUMN] = worker.get_parts_count(sf_data.file_name)

    def __check_file_convergence(self, stage: int, sf_data: FileData) -> None:
        """
        Delegates verification of equality w.r.t. current and desired_distributions to the inputed FileData instance
        :param int stage: number representing the discrete time step the simulation is currently at
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        """
        if sf_data.equal_distributions():
            print("Singular convergence at stage {}".format(stage))
            sf_data.convergence_data.cswc_increment(1)
            sf_data.convergence_data.try_append_to_convergence_set(stage)
        else:
            print("Cswc reset at stage {}".format(stage))
            sf_data.convergence_data.save_sets_and_reset()
    # endregion

    # region teardown
    def __workers_stop_tracking_shared_file(self, sf_data: FileData) -> None:
        """
        Tears down the hive responsible of persisting the inputed shared file by removing all* references to it in both
        the Hivemind aswell as the workers who were sharing its parts.
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        """
        sf_data.fclose()
        hive_workers_names: List[str] = [*sf_data.desired_distribution.index]
        sf_name: str = str_copy(sf_data.file_name)  # Creates an hard copy (str) of the shared file name
        # First ask workers to reset theirs, for safety, popping in hivemind structures is only done at a later point
        self.__remove_routing_tables(sf_name, hive_workers_names)

    def __hivemind_stops_tracking_shared_files(self, sf_names: List[str]) -> None:
        """
        :param List[str] sf_names: shared file names that won't ever again be recoverable due to a worker's disconnect
        """
        for sf_name in sf_names:
            try:
                self.shared_files.pop(sf_name)
                self.sf_data.pop(sf_name)
            except KeyError as kE:
                log.error("Key ({}) doesn't exist in hivemind's shared_files or sf_data dictionaries".format(sf_name))
                log.error("Key Error message: {}".format(str(kE)))
    # endregion

    # region hive recovery methods
    def __heal_hive(self, sf_data: FileData, dead_worker: Worker) -> Dict[str, str]:
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param FileData sf_data: reference to FileData instance object whose fields need to be updated
        :param Worker dead_worker: The worker to be droppedd from desired distribution, etc...
        :returns Dict[str, str] replacement_dict: if no viable replacement was found, the dict will be empty
        """
        labels: List[str]
        replacing_worker: Optional[Worker]
        replacement_dict: Dict[str, str]
        labels, replacing_worker, replacement_dict = self.__find_replacement_node(sf_data, dead_worker.name)

        if replacement_dict:
            sf_data.commit_heal(replacement_dict)
            self.__inherit_routing_table(sf_data.file_name, dead_worker, replacing_worker, replacement_dict)
            self.__update_routing_tables(sf_data.file_name, labels, dead_worker, replacement_dict)
        return replacement_dict

    def __find_replacement_node(self, sf_data: FileData, dw_name: str) -> Tuple[List[str], Optional[Worker], Dict[str, str]]:
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param FileData sf_data: reference to FileData instance object whose fields need to be updatedd
        :param str dw_name: name of the worker to be dropped from desired distribution, etc...
        :returns Tuple[List[str], Optional[Worker], Dict[str, str]]: key pair of dead_worker_name : replacement_worker_name or None
        """
        labels = [*sf_data.desired_distribution.index]
        dict_items = self.workers_uptime.items()

        if len(labels) == len(dict_items):
            return [], None, {}  # before a worker's disconnection the hive already had all existing network workers

        base_uptime = self.workers_uptime[dw_name] - 1.0
        while base_uptime is not None:
            for rw_name, uptime in dict_items:
                if self.worker_status[rw_name] == Status.ONLINE and rw_name not in labels and uptime > base_uptime:
                    return labels, self.workers[rw_name], {dw_name: rw_name}
            base_uptime = self.__expand_uptime_range_search(base_uptime)

        return [], None, {}  # no replacement was found, all possible replacements seem to be offline or suspected

    def __contract_hive(self, sf_data: FileData, dw_name: str):
        cropped_adj_matrix = self.__crop_adj_matrix(sf_data, dw_name)
        cropped_labels, cropped_ddv = self.__crop_desired_distribution(sf_data, dw_name)
        transition_matrix = mh.metropolis_algorithm(cropped_adj_matrix, cropped_ddv, column_major_out=True)
        transition_matrix = pd.DataFrame(transition_matrix, index=cropped_labels, columns=cropped_labels)
        sf_data.reset_adjacency_matrix(cropped_labels, cropped_adj_matrix)
        sf_data.reset_distribution_data(cropped_labels, cropped_ddv)
        sf_data.reset_density_data()
        sf_data.reset_convergence_data()
        self.__set_routing_tables(sf_data.file_name, cropped_labels, transition_matrix)

    def __crop_adj_matrix(self, sf_data: FileData, dw_name: str):
        """
        Generates a new desired distribution vector which is a subset of the original one.
        :param sf_data: reference to FileData instance object whose fields need to be updated
        :type domain.helpers.FileData
        :param dw_name: name of the worker to be dropped from desired distribution, etc...
        :type str
        :return: surviving worker names and their new desired density
        :rtype list<list<int>>
        """
        sf_data.desired_distribution.drop(dw_name, axis='index', inplace=True)
        sf_data.desired_distribution.drop(dw_name, axis='columns', inplace=True)
        # TODO After the first failure we must update the ADJ Matrix labels, that is causing an error
        size = sf_data.desired_distribution.shape[0]
        for i in range(size):
            is_absorbent_or_transient = True
            for j in range(size):
                # Ensure state i can reach and be reached by some other state j, where i != j
                if sf_data.desired_distribution.iat[i, j] == 1 and i != j:
                    is_absorbent_or_transient = False
                    break
            if is_absorbent_or_transient:
                j = Hivemind.random_j_index(i, size)
                sf_data.desired_distribution.iat[i, j] = 1
                sf_data.desired_distribution.iat[j, i] = 1
        return sf_data.desired_distribution.values.tolist()

    def __crop_desired_distribution(self, sf_data: FileData, w_name: str):
        """
        Generates a new desired distribution vector which is a subset of the original one.
        :param sf_data: reference to FileData instance object whose fields need to be updatedd
        :type domain.helpers.FileData
        :param w_name: name of the worker to be dropped from desired distribution, etc...
        :type str
        :return: surviving worker names and their new desired density
        :rtype list<string>, list<float>
        """
        # get probability dead worker's density, 'share it' by remaining workers, then remove it from vector column
        increment = \
            sf_data.desired_distribution.at[w_name, DEFAULT_COLUMN] / (sf_data.desired_distribution.shape[0] - 1)

        sf_data.desired_distribution.drop(w_name, inplace=True)
        # fetch remaining labels and rows as found on the dataframe
        new_labels = [*sf_data.desired_distribution.index]
        new_values = [*sf_data.desired_distribution.iloc[:, 0]]
        incremented_values = [value + increment for value in new_values]
        return new_labels, incremented_values
    # endregion

    # region other helpers
    def __set_routing_tables(self, sf_name: str, worker_names: List[str], transition_matrix: pd.DataFrame):
        """
        Inits the routing tables w.r.t. the inputed shared file name for all listed workers' names by passing them their
        respective vector column within the transition_matrix
        :param str sf_name: name of the shared file
        :param List[str] worker_names: name of the workers that share the file
        :param pd.DataFrame transition_matrix: labeled transition matrix to be splitted between named workers
        """
        for name in worker_names:
            transition_vector = transition_matrix.loc[:, name]  # <label, value> pairs in column[worker_name]
            self.workers[name].set_file_routing(sf_name, transition_vector)

    def __inherit_routing_table(self, sf_name: str, dead_worker: Worker, new_worker: Worker, replacement_dict: Dict[str, str]) -> None:
        """
        The new Worker instance receives the transition vector, of a shared file, of the disconnected Worker instance.
        The inheritance involves renaming a row within the column vector.
        :param str sf_name: name of the file whose routing table is going to be passed between the workers
        :param Worker dead_worker: Worker instance recently disconnected from the an hive
        :param Worker new_worker: Worker instance that is replacing him in that hive
        :param Dict[str, str] replacement_dict: old worker name, new worker name)
        """
        new_worker.set_file_routing(sf_name, dead_worker.routing_table[sf_name].rename(index=replacement_dict))

    def __update_routing_tables(self, sf_name: str, worker_names: List[str], dead_worker: Worker, replacement_dict: Dict[str, str]) -> None:
        """
        Updates the routing tables w.r.t. the inputed shared file name for all listed workers' names without altering
        their transition_vectors. After ensuring the
        :param str sf_name: name of the shared file
        :param List[str] worker_names: name of the workers that share the file
        :param Dict[str, str] replacement_dict: old worker name, new worker name)
        """
        workers_names_needing_update = self.__try_remove_worker_from_list(worker_names, dead_worker.name)
        for name in workers_names_needing_update:
            self.workers[name].update_file_routing(sf_name, replacement_dict)

    def __remove_routing_tables(self, sf_name: str, worker_names: List[str]) -> None:
        """
        :param str sf_name: name of the shared file to be removed from workers' routing tables
        :param List[str] worker_names: name of the workers that share the file
        """
        for name in worker_names:
            self.workers[name].remove_file_routing(sf_name)

    def __filter_and_map_online_workers(self) -> List[Worker]:
        """
        Filters and maps the workers' being managed by the Hivemind instance according to their Online statuses.
        :returns List[Worker]: objects whose status is online
        """
        return [*map(lambda w: w[0], [*filter(lambda i: i[1] == Status.ONLINE, self.worker_status.items())])]

    def __expand_uptime_range_search(self, current_uptime: float) -> Optional[float]:
        """
        :param float current_uptime: current acceptance criteria for a replacement node
        :returns float: current_uptime - 10.0f, 0.0f or None
        """
        if current_uptime == 0.0:
            return None
        current_uptime -= 10.0
        return current_uptime if current_uptime > 50.0 else 0.0

    def __try_remove_worker_from_list(self, worker_name_list: List[str], w_name: str) -> List[str]:
        """
        :param List[str] worker_name_list: list of workers' names that belonged to an hive
        :param str w_name: name of the worker to remove from the given list if possible
        :returns List[str] worker_name_list: unmodified or without the specified worker_name
        """
        try:
            worker_name_list.remove(w_name)
            return worker_name_list
        except ValueError:
            return worker_name_list
    # endregion
    # endregion

    # region static methods
    @staticmethod
    def random_j_index(i: int, size: int):
        """
        Returns a random index j, that is between [0, size) and is different than i
        :param int i: an index
        :param int size: the size of the matrix
        :returns int j
        """
        size_minus_one = size - 1
        if i == 0:
            return random.randrange(start=1, stop=size)  # any node j other than the first (0)
        elif i == size_minus_one:
            return random.randrange(start=0, stop=size_minus_one)  # any node j except than the last (size-1)
        elif 0 < i < size_minus_one:
            return excluding_randrange(start=0, stop=i, start_again=(i + 1), stop_again=size)
    # endregion
