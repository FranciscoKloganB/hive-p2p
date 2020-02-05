import json
import os
from typing import List, Union, Dict, Tuple, Optional, Any

from domain.Enums import Status
from domain.Hive import Hive
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.helpers.FileData import FileData
from globals.globals import SHARED_ROOT, SIMULATION_ROOT, READ_SIZE, DEFAULT_COLUMN
from utils.collections import safe_remove


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
                    hive.file.parts_count = part_number

            # Distribute files before starting simulation
            for hive in self.hives.values():
                hive.spread_files(files_spreads[hive.file.name], files_dict[hive.file.name])
    # endregion

    # region Simulation Interface
    def execute_simulation(self) -> None:
        """
        Runs a stochastic swarm guidance algorithm applied to a P2P network
        """
        for stage in range(self.max_epochs):
            for hive in self.hives.values():
                hive.execute_epoch()

            self.__process_stage_results(stage)

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
        raise NotImplementedError()
    # endregion

    # region Stage Processing
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
                sf_data.current_distribution.at[id, DEFAULT_COLUMN] = worker.get_file_parts_count(sf_data.name)

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

    # region Peer Search and Cloud References
    def find_replacement_worker(self, sf_data: FileData, dw_name: str) -> Tuple[List[str], Optional[Worker], Dict[str, str]]:
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
            base_uptime = self.relax_replacement_conditions(base_uptime)

        return [], None, {}  # no replacement was found, all possible replacements seem to be offline or suspected

    def relax_replacement_conditions(self, current_uptime: float) -> Optional[float]:
        """
        Decreases the minimum uptime required to accept a worker as replacement of some other disconnected worker
        :param float current_uptime: current acceptance criteria for a replacement node
        :returns Optional[float]: current_uptime - 10.0f, 0.0f or None
        """
        if current_uptime == 0.0:
            return None
        current_uptime -= 10.0
        return current_uptime if current_uptime > 50.0 else 0.0

    def get_cloud_reference(self) -> str:
        """
        TODO: future-iteration
        :returns a cloud reference that can be used to persist files with more reliability
        """
        return ""
    # endregion

    # region Helpers
    def __new_hive(self, shared: Dict[str, Dict[str, Union[List[str], str]]], file_name: str) -> Hive:
        """
        Creates a new hive
        """
        hive_members: Dict[str, Worker] = {}
        for worker_id in shared[file_name]['members']:
            hive_members[worker_id] = self.workers[worker_id]
        hive = Hive(self, file_name, hive_members)
        self.hives[hive.id] = hive
        return hive

    # endregion
