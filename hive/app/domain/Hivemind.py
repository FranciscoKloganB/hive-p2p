import os
import json
import random
import logging as log
import numpy as np
import pandas as pd
import domain.metropolis_hastings as mh

from pathlib import Path
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.Enums import Status, HttpCodes
from domain.helpers.FileData import FileData
from domain.helpers.ConvergenceData import ConvergenceData
from globals.globals import SHARED_ROOT, READ_SIZE, DEFAULT_COLUMN
from utils.convertions import str_copy
from utils.randoms import excluding_randrange

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
    __STAGES_WITH_CONVERGENCE = []
    __MAX_CONSECUTIVE_CONVERGENCE_STAGES = 0

    def __init__(self, simfile_name):
        """
        :param simfile_name: path to json file containing the parameters this simulation should execute with
        :type str
        """
        with open(simfile_name) as input_file:
            json_obj = json.load(input_file)
            # Init basic simulation variables
            self.shared_files = {}
            self.sf_data = {}
            self.workers = {}
            self.worker_status = {}
            self.workers_uptime = json_obj['nodes_uptime']
            self.max_stages = json_obj['max_simulation_stages']
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
                np.random.choice(choices).receive_part(part)
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
        file_name = Path(file_path).resolve().stem

        with open(file_path, "rb") as shared_file:
            file_parts = {}
            part_number = 0
            while True:
                read_buffer = shared_file.read(READ_SIZE)
                if read_buffer:
                    part_number = part_number + 1
                    shared_file_part = SharedFilePart(
                        file_name,
                        part_number,
                        read_buffer
                    )
                    file_parts[part_number] = shared_file_part
                else:
                    # Keeps track of how many parts the file was divided into
                    self.shared_files[file_name] = file_parts
                    break
            self.sf_data[file_name] = FileData(file_name=file_name, parts_count=part_number)
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
            surviving_workers = self.__remove_some_workers(online_workers_list, stage)
            for worker in surviving_workers:
                worker.do_stage()
            self.__process_stage_results(stage)

    def __redistribute_transition_matrices(self, shared_file_names, dead_worker_name):
        for sf_name in shared_file_names:
            self.__care_taking(self.sf_data[sf_name], dead_worker_name)

    def __care_taking(self, sf_data, dw_name):
        """
        :param sf_data: data class containing generalized information regarding the shared file
        :type domain.helpers.FileData
        :param dw_name: name of the worker that left the hive willingly or unexpectedly
        :type str
        """
        if self.__heal_hive(sf_data, dw_name) is None:
            self.__contract_hive(sf_data, dw_name)  # contraction only occurs if healing fails

    def __init_recovery_protocol(self, sf_data, mock=None):
        """
        Starts file recovering by asking the best in the Hive still alive to run a file reconstruction algorithm
        :param sf_data: instance object containing information about the file to be recovered
        :type domain.helpers.FileData
        :param mock: allows simulation to do a recovery by passing files from dead worker to a living worker
        :type dict<int, domain.SharedFilePart>
        """
        best_worker_name = sf_data.highest_density_node_label

        if self.worker_status[best_worker_name] != Status.ONLINE:
            # TODO future-iterations:
            #  get the next best node, until a node in the hive is found to be online or no other options remain
            #  for current iteration, assume this situation doesn't happen, best node in the structure is always online
            return

        worker = self.workers[best_worker_name]
        if mock:
            worker.receive_parts(sf_id_parts=mock, sf_name=sf_data.file_name, no_check=True)
        else:
            worker.init_recovery_protocol(sf_data.file_name)

    def receive_complaint(self, suspects_name, sf_name=None):
        # TODO future-iterations:
        #  1. register complaint
        #  2. when byzantine complaints > threshold
        #    2.1. find away of obtaining shared_file_names user had
        #    2.2. discover the files the node used to share, probably requires yet another sf_strucutre
        #    2.3. ask the next highest density node that is alive to rebuild dead nodes' files
        if sf_name and suspects_name:
            pass  # maybe sf_name can help when enough complaints are received, reanalyze this param at a later date.
        log.warning("receive_complaint is only a mock. method needs to be implemented...")

    def route_file_part(self, dest_worker, part):
        """
        :param dest_worker: destinatary of the file part
        :type domain.Worker OR str (domain.Worker.name)
        :param part: the file part to send to specified worker
        :type domain.SharedFilePart
        :returns: http codes based status of destination worker
        :rtype int
        """
        dest_status = self.worker_status[dest_worker]
        if dest_status == Status.ONLINE:
            dest_worker.receive_part(part)
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
    def __init_file_data(self, sf_name, adj_matrix, desired_distribution, labels):
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
    def __remove_some_workers(self, online_workers, stage=None):
        """
        For each online worker, if they are online, see if they remain alive for the next stage or if they die,
        according to their uptime record.
        :param online_workers: collection of workers that are known to be online
        :type list<domain.Worker>
        :returns surviving_workers: subset of online_workers, who weren't selected to be removed from the hive
        :rtype list<domain.Worker>
        """
        surviving_workers = []
        for worker in online_workers:
            uptime = self.workers_uptime[worker.name] / 100
            remains_alive = np.random.choice([True, False], p=[uptime, 1 - uptime])
            if remains_alive:
                surviving_workers.append(worker)
            else:
                self.__remove_worker(worker, stage=stage)
        return surviving_workers

    def __remove_worker(self, target, stage=None):
        """
        :param target: worker who is going to be removed from the simulation network
        :type domain.Worker
        """
        sf_parts = target.get_all_parts()
        self.worker_status[str_copy(target.name)] = Status.OFFLINE

        for sf_name, sfp_id in sf_parts.items():
            sf_data = self.sf_data[sf_name]
            parts_count = len(sfp_id)
            threshold = sf_data.get_failure_threshold()
            if parts_count > threshold:
                # Can't recover from this situation, stop tracking file and ask other sharers to do the same
                self.__register_failure(stage, sf_name, target, parts_count, threshold)
                self.__stop_tracking_shared_file(sf_name)
            else:
                # Replace dead node with similar one, or, do hive contraction
                self.__care_taking(sf_data, target)
                # Ask best node still alive to do recovery protocols, assume perfect failure detector
                self.__init_recovery_protocol(sf_data, mock=sf_parts)
        del target  # mark as ready for garbage collection - no longer need the worker instance

    def __process_stage_results(self, stage):
        """
        For each file being shared on this hivemind network, check if its desired distribution has been achieved.
        :param stage: stage number - the one that is being processed
        :type int
        """
        if stage == self.max_stages:
            exit(0)
        for sf_data in self.sf_data.values():
            # retrieve from each worker how many parts they have for current data.file_name and update convergence data
            self.__request_file_counts(sf_data)
            # when all queries for a file are done, verify convergence for data.file_name
            self.__check_file_convergence(sf_data, stage)

    def __request_file_counts(self, sf_data):
        worker_names = [*sf_data.desired_distribution.index]
        for name in worker_names:
            if self.worker_status[name] != Status.ONLINE:
                sf_data.current_distribution.at[name, DEFAULT_COLUMN] = 0
            else:
                worker = self.workers[name]  # get worker instance corresponding to name
                sf_data.current_distribution.at[name, DEFAULT_COLUMN] = worker.request_file_count(sf_data.file_name)

    def __check_file_convergence(self, sf_data, stage):
        if sf_data.equal_distributions():
            sf_data.convergence_data.cswc_increment_and_get(1)
            if sf_data.convergence_data.try_update_convergence_set(stage):
                with open("outfile.txt", "a+") as out_file:
                    out_file.write("File {} converged at stage {}...\n"
                                   "Desired distribution: {},\n"
                                   "Current distribution: {}\n".format(sf_data.file_name,
                                                                       stage,
                                                                       sf_data.desired_distribution.to_string(),
                                                                       sf_data.current_distribution.to_string()
                                                                       )
                                   )
        else:
            sf_data.convergence_data.save_sets_and_reset()
    # endregion

    # region teardown
    def __stop_tracking_shared_file(self,  sf_name):
        labels = [*self.sf_data[sf_name].desired_distribution.index]
        # First reset hivemind data structures
        self.sf_data.pop(sf_name, None)
        self.shared_files.pop(sf_name, None)
        # Now ask workers to reset theirs
        self.__remove_routing_tables(sf_name, labels)
    # endregion

    # region hive recovery methods
    def __heal_hive(self, sf_data, dw_name):
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param sf_data: reference to FileData instance object whose fields need to be updatedd
        :type domain.helpers.FileData
        :param dw_name: name of the worker to be droppedd from desired distribution, etc...
        :type str
        :returns: None if hive is unable to heal through replacement, or a replacement dict, if it was
        :rtype dict<str, str>
        """
        labels, replacement_dict = self.__find_replacement_node(sf_data, dw_name)
        if replacement_dict:
            sf_data.replace_distribution_node(replacement_dict)
            sf_data.reset_density_data()
            sf_data.reset_convergence_data()
            self.__update_routing_tables(sf_data.file_name, labels, replacement_dict)
        return replacement_dict

    def __find_replacement_node(self, sf_data, dw_name):
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param sf_data: reference to FileData instance object whose fields need to be updatedd
        :type domain.helpers.FileData
        :param dw_name: name of the worker to be droppedd from desired distribution, etc...
        :type str
        :return: key pair of dead_worker_name : replacement_worker_name or None
        :rtype dict<str, str>
        """
        labels = [*sf_data.desired_distribution.index]
        dict_items = self.workers_uptime.items()

        if len(labels) == len(dict_items):
            return None

        base_uptime = self.workers_uptime[dw_name] - 10.0
        while base_uptime is not None:
            for name, uptime in dict_items:
                if self.worker_status[name] == Status.ONLINE and name not in labels and uptime > base_uptime:
                    return labels, {dw_name: name}
            base_uptime = self.__expand_uptime_range_search(base_uptime)
        return None

    def __contract_hive(self, sf_data, dw_name):
        cropped_adj_matrix = self.__crop_adj_matrix(sf_data, dw_name)
        cropped_labels, cropped_ddv = self.__crop_desired_distribution(sf_data, dw_name)
        transition_matrix = mh.metropolis_algorithm(cropped_adj_matrix, cropped_ddv, column_major_out=True)
        transition_matrix = pd.DataFrame(transition_matrix, index=cropped_labels, columns=cropped_labels)
        sf_data.reset_adjacency_matrix(cropped_adj_matrix)
        sf_data.reset_distribution_data(cropped_labels, cropped_ddv)
        sf_data.reset_density_data()
        sf_data.reset_convergence_data()
        self.__set_routing_tables(sf_data.file_name, cropped_labels, transition_matrix)

    def __crop_adj_matrix(self, sf_data, dw_name):
        """
        Generates a new desired distribution vector which is a subset of the original one.
        :param sf_data: reference to FileData instance object whose fields need to be updatedd
        :type domain.helpers.FileData
        :param dw_name: name of the worker to be dropped from desired distribution, etc...
        :type str
        :return: surviving worker names and their new desired density
        :rtype list<list<int>>
        """
        sf_data.desired_distribution.drop(dw_name, axis='index', inplace=True)
        sf_data.desired_distribution.drop(dw_name, axis='columns', inplace=True)
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

    def __crop_desired_distribution(self, sf_data, dw_name):
        """
        Generates a new desired distribution vector which is a subset of the original one.
        :param sf_data: reference to FileData instance object whose fields need to be updatedd
        :type domain.helpers.FileData
        :param dw_name: name of the worker to be dropped from desired distribution, etc...
        :type str
        :return: surviving worker names and their new desired density
        :rtype list<string>, list<float>
        """
        # get probability dead worker's density, 'share it' by remaining workers, then remove it from vector column
        increment = \
            sf_data.desired_distribution.at[dw_name, DEFAULT_COLUMN] / (sf_data.desired_distribution.shape[0] - 1)

        sf_data.desired_distribution.drop(dw_name, inplace=True)
        # fetch remaining labels and rows as found on the dataframe
        new_labels = [*sf_data.desired_distribution.index]
        new_values = [*sf_data.desired_distribution.iloc[:, 0]]
        incremented_values = [value + increment for value in new_values]
        return new_labels, incremented_values
    # endregion

    # region other helpers
    def __set_routing_tables(self, sf_name, worker_names, transition_matrix):
        """
        :param sf_name: name of the shared file
        :type str
        :param worker_names: name of the workers that share the file
        :type list<str>
        :param transition_matrix: labeled transition matrix to be splitted between named workers
        :type N-D pandas.DataFrame
        """
        for name in worker_names:
            transition_vector = transition_matrix.loc[:, name]  # <label, value> pairs in column[worker_name]
            self.workers[name].set_file_routing(sf_name, transition_vector)

    def __update_routing_tables(self, sf_name, worker_names, replacement_dict):
        """
        :param sf_name: name of the shared file
        :type str
        :param worker_names: name of the workers that share the file
        :type list<str>
        :param replacement_dict: key, value pair where key represents the name to be replaced with the new value
        :type dict<str, str>
        """
        for name in worker_names:
            self.workers[name].update_file_routing(sf_name, replacement_dict)

    def __remove_routing_tables(self, sf_name, worker_names):
        """
        :param sf_name: name of the shared file to be removed from workers' routing tables
        :type str
        :param worker_names: name of the workers that share the file
        :type list<str>
        """
        for name in worker_names:
            self.workers[name].remove_file_routing(sf_name)

    def __filter_and_map_online_workers(self):
        """
        Selects workers (w[0] := worker_status.keys()) who are online (i[1] := worker_status.values())
        :returns Workers objects whose status is online
        :rtype list<domain.Worker>
        """
        return [*map(lambda w: w[0], [*filter(lambda i: i[1] == Status.ONLINE, self.worker_status.items())])]

    def __expand_uptime_range_search(self, current_uptime):
        if current_uptime == 0.0:
            return None
        current_uptime -= 10.0
        if current_uptime > 50.0:
            return current_uptime
        return 0.0

    def __register_failure(self, stage, sf_name, target, parts_count, threshold):
        with open("out_{}.txt".format(sf_name), "a+") as out_file:
            out_file.write("Failure at stage {} due to worker '{}' death, parts: {} > threshold: {}\n"
                           .format(stage, target, parts_count, threshold)
                           )
    # endregion
    # endregion

    # region static methods
    @staticmethod
    def random_j_index(i, size):
        size_minus_one = size - 1
        if i == 0:
            return random.randrange(start=1, stop=size)  # any node j other than the first (0)
        elif i == size_minus_one:
            return random.randrange(start=0, stop=size_minus_one)  # any node j except than the last (size-1)
        elif 0 < i < size_minus_one:
            return excluding_randrange(start=0, stop=i, start_again=(i + 1), stop_again=size)
    # endregion
