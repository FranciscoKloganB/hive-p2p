import os
import json
import numpy as np
import pandas as pd
import utils.metropolis_hastings as mh
import logging as log

from pathlib import Path

from domain.Hive import Hive
from domain.Enums import Status
from domain.Worker import Worker
from domain.helpers.FileData import FileData
from domain.SharedFilePart import SharedFilePart
from domain.helpers.ConvergenceData import ConvergenceData

from utils.randoms import random_index
from utils.convertions import str_copy
from utils.collections import safe_remove

from globals.globals import SHARED_ROOT, SIMULATION_ROOT, READ_SIZE, DEFAULT_COLUMN

from typing import cast, List, Set, Union, Dict, Tuple, Optional, Any


class Hivemind:
    """
    Representation of the P2P Network super node managing one or more hives ---- Simulator is piggybacked
    :ivar int max_epochs: number of stages the hive has to converge to the ddv before simulation is considered failed
    :ivar Dict[str, Hive] hives: collection mapping hives' uuid (attribute Hive.id) to the Hive instances
    :ivar Dict[str, Worker] workers: collection mapping workers' names to their Worker instances
    :ivar Dict[str, FileData] files_data: collection mapping file names on the system and their FileData instance
    :ivar Dict[Union[Worker, str], int] workers_status: maps workers or their names to their connectivity status
    :ivar Dict[str, List[FileData]] workers_hives: maps workers' names to hives they are known to belong to
    :ivar Dict[str, float] workers_uptime: maps workers' names to their expected uptime
    """

    # region instance variables and constructors
    def __init__(self, simfile_name: str) -> None:
        """
        Instantiates an Hivemind object
        :param str simfile_name: path to json file containing the parameters this simulation should execute with
        """
        simfile_path: str = os.path.join(SIMULATION_ROOT, simfile_name)
        with open(simfile_path) as input_file:
            json_obj: Any = json.load(input_file)

            # Init basic simulation variables
            self.max_epochs: int = json_obj['max_epochs']
            self.hives: Dict[str, Hive] = {}
            self.workers: Dict[str, Worker] = {}

            # Instantiaite jobless Workers
            for worker_id, worker_uptime in json_obj['peers_uptime'].items():
                worker: Worker = Worker(worker_id, worker_uptime)
                self.workers[worker.id] = worker

            # Read and split all shareable files specified on the input, also assign Hive initial attributes (uuid, members, and FileData)
            hive: Hive
            hive_members: Dict[str, Worker]
            files_spreads: Dict[str, str] = {}
            files_dict: Dict[str, Dict[int, SharedFilePart]] = {}
            file_parts: Dict[int, SharedFilePart]

            shared: Dict[str, Dict[str, Union[List[str], str]]] = json_obj['shared'].keys()
            for file_name in shared.keys():
                with open(os.path.join(SHARED_ROOT, file_name), "rb") as file:
                    part_number: int = 0
                    file_parts = {}
                    files_spreads[file_name] = shared[file_name]['spread']
                    hive = self.__new_hive(shared, file_name)  # Among other things, assigns initial Hive members to the instance, implicitly set routing tables
                    while True:
                        read_buffer = file.read(READ_SIZE)
                        if read_buffer:
                            part_number = part_number + 1
                            file_parts[part_number] = SharedFilePart(hive.id, file_name, part_number, read_buffer)
                        else:
                            files_dict[file_name] = file_parts
                            break
                    hive.file = FileData(name=file_name, parts_count=part_number)

            # Distribute files before starting simulation
            for hive in self.hives.values():
                hive.spread_files(files_spreads[hive.file.name], files_dict[hive.file.name])
    # endregion

    # region simulation execution methods
    def execute_simulation(self) -> None:
        """
        Runs a stochastic swarm guidance algorithm applied to a P2P network
        """
        online_workers_list: List[Worker] = self.__filter_and_map_online_workers()
        for stage in range(self.max_epochs):
            online_workers_list = self.__remove_some_workers(online_workers_list, stage)
            for worker in online_workers_list:
                worker.execute_epoch()
            self.__process_stage_results(stage)

    def __care_taking(self, stage: int, sf_data: FileData, dead_worker: Worker) -> bool:
        """
        Cleans up after a worker disconnection by healing up or shrinking or failing!
        :param int stage: number representing the discrete time step the simulation is currently at
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        :param Worker dead_worker: instance object corresponding to the worker who left the network
        """
        sf_data.fwrite("Initializing care taking process at stage {} due to worker '{}' death...".format(stage, dead_worker.id))
        if self.__heal_hive(sf_data, dead_worker):
            sf_data.fwrite("Heal complete!")
            return True

        sf_data.fwrite("Hive healing was not possible...")
        if self.__shrink_hive(sf_data, dead_worker.id):
            sf_data.fwrite("Shrinking complete!")
            return True  # successful hive shrinking

        sf_data.fwrite("Care taking couldn't recover from worker failure...".format(stage))
        return False

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
            worker.receive_parts(mock, sf_data.name, no_check=True)
        else:
            sf_data.fwrite("Asking best worker, {}, to initialize recovery protocols...".format(best_worker_name))
            worker.init_recovery_protocol(sf_data.name)
            print("   Recovery complete...")

    def receive_complaint(self, suspects_name: str) -> None:
        """
        Registers a complaint on the named worker, if enough complaints are received, broadcasts proper action to all
        hives' workers to which the suspect belonged to.
        :param suspects_name: id of the worker which regards the complaint
        """
        # TODO future-iterations:
        #  1. register complaint
        #  2. when byzantine complaints > threshold
        #    2.1. find away of obtaining shared_file_names user had
        #    2.2. discover the files the node used to share, probably requires yet another sf_strucutre
        #    2.3. ask the next highest density node that is alive to rebuild dead nodes' files
        log.warning("receive_complaint for {} is only a mock. method needs to be implemented...".format(suspects_name))
    # endregion

    # region helper methods
    # region setup
    def __init_file_data(
            self, sf_name: str, adj_matrix: List[List[int]], desired_distribution: List[float], labels: List[str]) -> None:
        """
        Creates a FileData instance and maps it to file name in the Hivemind's instance sf_data field dictionary
        :param str sf_name: the name of the file to be tracked by the hivemind
        :param List[List[int]] adj_matrix: matrix with connections between worker nodes
        :param List[float] desired_distribution: the desired distribution vector of the given named file
        :param List[str] labels: id of the workers belonging to the hive, i.e.: keepers or sharers of the files
        """
        sf_data = self.files_data[sf_name]
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
        :param List[Worker] online_workers: collection of workers that are known to be online
        :returns List[Worker] surviving_workers: subset of online_workers, who weren't selected to be removed from the hive
        """
        surviving_workers: List[Worker] = []
        for worker in online_workers:
            uptime: float = self.workers_uptime[worker.id] / 100
            remains_alive: bool = np.random.choice(Hivemind.TRUE_FALSE, p=[uptime, 1 - uptime])
            if remains_alive:
                surviving_workers.append(worker)
            else:
                self.__remove_worker(worker, stage=stage)
        return surviving_workers

    def __remove_worker(self, dead_worker: Worker, stage: int = None) -> None:
        """
        Changes a Worker's status to Status.OFFLINE, it then register a file persistence failure or recovers accordingly
        :param Worker dead_worker: Worker instance to be removed to be forcefully disconnected
        :param int stage: number representing the discrete time step the simulation is currently at
        """
        sf_data: FileData
        sf_failures: Set[str] = set()
        shared_files: Dict[str, Dict[int, SharedFilePart]] = dead_worker.get_all_parts()
        if not shared_files:  # if dead worker had no shared files on him just try to replace node or shrink the hive
            for sf_data in self.workers_hives[dead_worker.id]:
                sf_data.fwrite("Worker: '{}' was removed at stage {}, he had no files.".format(dead_worker.id, stage))
                sf_failures = self.__try_care_taking(stage, dead_worker, sf_data, sf_failures)
        else:  # otherwise see if a failure has happened before doing anything else
            for sf_name, sf_id_sfp_dict in shared_files.items():
                sf_data = self.files_data[sf_name]
                sf_data.fwrite("Worker: '{}' was removed at stage {}, he had {} parts of file {}".format(dead_worker.id, stage, len(sf_id_sfp_dict), sf_name))
                if len(sf_id_sfp_dict) > sf_data.get_failure_threshold():
                    self.__workers_stop_tracking_shared_file(sf_data)
                    sf_data.fclose("Worker had too many parts... file lost!")
                    sf_failures.add(sf_name)
                else:
                    sf_failures = self.__try_care_taking(stage, dead_worker, sf_data, sf_failures, recover=sf_id_sfp_dict)
        self.__stop_tracking_failed_hives(sf_failures)
        self.__stop_tracking_worker(dead_worker.id)
        self.workers_status[dead_worker.id] = Status.OFFLINE

    def __process_stage_results(self, stage: int) -> None:
        """
        Obtains all workers' densities regarding each shared file and logs progress in the system accordingly
        :param int stage: number representing the discrete time step the simulation is currently at
        """
        if stage == self.max_epochs - 1:
            for sf_data in self.files_data.values():
                sf_data.fwrite("\nReached final stage... Executing tear down processes. Summary below:")
                sf_data.convergence_data.save_sets_and_reset()
                sf_data.fwrite(str(sf_data.convergence_data))
                sf_data.fclose()
            exit(0)
        else:
            for sf_data in self.files_data.values():
                sf_data.fwrite("\nStage {}".format(stage))
                # retrieve from each worker their part counts for current sf_name and update convergence data
                self.__request_file_counts(sf_data)
                # when all queries for a file are done, verify convergence for data.file_name
                self.__check_file_convergence(stage, sf_data)

    def __request_file_counts(self, sf_data: FileData) -> None:
        """
        Updates inputted FileData.current_distribution with current density values for each worker sharing the file
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        """
        worker_ids = [*sf_data.desired_distribution.index]
        for id in worker_ids:
            if self.workers_status[id] != Status.ONLINE:
                sf_data.current_distribution.at[id, DEFAULT_COLUMN] = 0
            else:
                worker = self.workers[id]  # get worker instance corresponding to id
                sf_data.current_distribution.at[id, DEFAULT_COLUMN] = worker.get_parts_count(sf_data.name)

    def __check_file_convergence(self, stage: int, sf_data: FileData) -> None:
        """
        Delegates verification of equality w.r.t. current and desired_distributions to the inputted FileData instance
        :param int stage: number representing the discrete time step the simulation is currently at
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        """
        if sf_data.equal_distributions():
            print("Singular convergence at stage {}".format(stage))
            sf_data.convergence_data.cswc_increment(1)
            sf_data.convergence_data.try_append_to_convergence_set(stage)
        else:
            sf_data.convergence_data.save_sets_and_reset()
    # endregion

    # region teardown
    def __workers_stop_tracking_shared_file(self, sf_data: FileData) -> None:
        """
        Tears down the hive responsible of persisting the inputted shared file by removing all* references to it in both
        the Hivemind as well as the workers who were sharing its parts.
        :param FileData sf_data: data class instance containing generalized information regarding a shared file
        """
        hive_workers_names: List[str] = [*sf_data.desired_distribution.index]
        sf_name: str = str_copy(sf_data.name)  # Creates an hard copy (str) of the shared file id
        # First ask workers to reset theirs, for safety, popping in hivemind structures is only done at a later point
        self.__remove_routing_tables(sf_name, hive_workers_names)

    def __stop_tracking_failed_hives(self, sf_names: Set[str]) -> None:
        """
        Removes all references to the named shared file from the Hivemind instance structures and fields.
        :param List[str] sf_names: shared file names that won't ever again be recoverable due to a worker's disconnect
        """
        for sf_name in sf_names:
            try:
                self.files_data.pop(sf_name)
            except KeyError as kE:
                log.error("Key ({}) doesn't exist in hivemind's or sf_data dictionaries".format(sf_name))
                log.error("Key Error message: {}".format(str(kE)))
    # endregion

    # region hive recovery methods
    def __heal_hive(self, sf_data: FileData, dead_worker: Worker) -> Dict[str, str]:
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param FileData sf_data: reference to FileData instance object whose fields need to be updated
        :param Worker dead_worker: The worker to be dropped from desired distribution, etc...
        :returns Dict[str, str] replacement_dict: if no viable replacement was found, the dict will be empty
        """
        labels: List[str]
        replacing_worker: Optional[Worker]
        replacement_dict: Dict[str, str]

        sf_data.fwrite("Attempting to find a node replacement...")
        labels, replacing_worker, replacement_dict = self.__find_replacement_node(sf_data, dead_worker.id)

        if replacement_dict:
            sf_data.fwrite("Committing the replacement of nodes: {}".format(str(replacement_dict)))
            sf_data.commit_replacement(replacement_dict)
            self.workers_hives[replacing_worker.id].add(sf_data)
            self.__inherit_routing_table(sf_data.name, dead_worker, replacing_worker, replacement_dict)
            self.__update_routing_tables(sf_data.name, labels, replacement_dict)
        else:
            sf_data.fwrite("No replacement found...")

        return replacement_dict

    def __find_replacement_node(self, sf_data: FileData, dw_name: str) -> Tuple[List[str], Optional[Worker], Dict[str, str]]:
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param FileData sf_data: reference to FileData instance object whose fields need to be updated
        :param str dw_name: id of the worker to be dropped from desired distribution, etc...
        :returns Tuple[List[str], Optional[Worker], Dict[str, str]]: labels, new_worker, old_worker_name:new_worker_name
        """
        labels: List[str] = [*sf_data.desired_distribution.index]
        dict_items = self.workers_uptime.items()
        if len(labels) == len(dict_items):
            return [], None, {}  # before a worker's disconnection the hive already had all existing network workers

        base_uptime: Optional[float] = self.workers_uptime[dw_name] - 1.0
        while base_uptime is not None:
            for rw_name, uptime in dict_items:
                if self.workers_status[rw_name] == Status.ONLINE and rw_name not in labels and uptime > base_uptime:
                    return safe_remove(labels, dw_name), self.workers[rw_name], {dw_name: rw_name}
            base_uptime = self.__expand_uptime_range_search(base_uptime)

        return [], None, {}  # no replacement was found, all possible replacements seem to be offline or suspected

    def __shrink_hive(self, sf_data: FileData, dw_name: str) -> bool:
        """
        Reduces the size of the hive by one, by removing all references to the dead worker in the hive FileData instance
        :param FileData sf_data: reference to FileData instance object whose fields need to be updated
        :param str dw_name: id of the worker to be dropped from desired distribution, etc...
        :returns bool: False if shrinking is not possible, True if shrinking was successful
        """
        cropped_adj_matrix: List[List[int]]
        cropped_labels: List[str]
        cropped_ddv: List[float]

        sf_data.fwrite("Attempting to shrink the hive...")
        if len(sf_data.desired_distribution) == 1:
            sf_data.fwrite("Hive only had one working node, further shrinking is impossible... File lost!")
            return False

        cropped_adj_matrix = self.__crop_adj_matrix(sf_data, dw_name)
        cropped_labels, cropped_ddv = self.__crop_desired_distribution(sf_data, dw_name)
        transition_matrix: np.ndarray = mh.metropolis_algorithm(cropped_adj_matrix, cropped_ddv, column_major_out=True)
        transition_matrix: pd.DataFrame = pd.DataFrame(transition_matrix, index=cropped_labels, columns=cropped_labels)
        sf_data.reset_adjacency_matrix(cropped_labels, cropped_adj_matrix)
        sf_data.reset_distribution_data(cropped_labels, cropped_ddv)
        sf_data.reset_density_data()
        sf_data.reset_convergence_data()
        self.__set_routing_tables(sf_data.name, cropped_labels, transition_matrix)
        return True

    # region Helpers
    def __new_hive(self, shared: Dict[str, Dict[str, Union[List[str], str]]], file_name: str) -> Hive:
        """
        Creates a new hive
        """
        hive_members = {}
        for worker_id in shared[file_name]['members']:
            hive_members[worker_id] = self.workers[worker_id]
        hive = Hive(hive_members)
        self.hives[hive.id] = hive
        return hive

    def __expand_uptime_range_search(self, current_uptime: float) -> Optional[float]:
        """
        Decreases the minimum uptime required to accept a worker as replacement of some other disconnected worker
        :param float current_uptime: current acceptance criteria for a replacement node
        :returns Optional[float]: current_uptime - 10.0f, 0.0f or None
        """
        if current_uptime == 0.0:
            return None
        current_uptime -= 10.0
        return current_uptime if current_uptime > 50.0 else 0.0

    def __try_care_taking(self, stage: int, dead_worker: Worker, sf_data: FileData, sf_failures: Set[str], recover: Dict[int, SharedFilePart] = None) -> Set[str]:
        """
        :param int stage: number representing the discrete time step the simulation is currently at
        :param Worker dead_worker: Worker instance that was removed from an hive
        :param FileData sf_data: reference to FileData instance object whose fields need to be updated
        :param Set[str] sf_failures: set of names that keep track of all failed hives
        :param Dict[int, SharedFilePart] recover: recovery will be accomplished by passing files from dead to living worker
        :returns Set[str] sf_failures: unmodified or with new failed hive names
        """
        if self.__care_taking(stage, sf_data, dead_worker):
            # TODO future-iterations:
            #  call actual recovery protocol by not using mock and removing mock from __try_care_taking method signature
            self.__init_recovery_protocol(sf_data, mock=recover) if recover else None
        else:
            sf_data.fclose()
            sf_failures.add(sf_data.name)
        return sf_failures
    # endregion

    # endregion
