import json
import os
import sys
from typing import List, Union, Dict, Any

from domain.Enums import Status
from domain.Hive import Hive
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.helpers.FileData import FileData
from globals.globals import SHARED_ROOT, SIMULATION_ROOT, READ_SIZE, DEFAULT_COLUMN, AVG_UPTIME


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

    # region Class Variables, Instance Variables and Constructors
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
        failed_hives: List[str] = []
        for stage in range(self.max_epochs):
            for hive in self.hives.values():
                if not hive.execute_epoch():
                    failed_hives.append(hive.id)
            for hive_id in failed_hives:
                self.hives.pop(hive_id)
                if not self.hives:
                    sys.exit("Simulation terminated at stage {} because all hives failed before max epochs were reached".format(stage))
    # endregion

    # region Keeper Interface
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
        for worker_id in worker_ids:
            if self.workers_status[worker_id] != Status.ONLINE:
                sf_data.current_distribution.at[worker_id, DEFAULT_COLUMN] = 0
            else:
                worker = self.workers[worker_id]  # get worker instance corresponding to id
                sf_data.current_distribution.at[worker_id, DEFAULT_COLUMN] = worker.get_file_parts_count(sf_data.name)

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
    def find_replacement_worker(self, exclusion_dict: Dict[str, Worker], quantity: int) -> Dict[str, Worker]:
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param Dict[str, Worker] exclusion_dict: collection of worker ids that the calling hive haves no interest in, for any reason
        :param int quantity: how many replacements the calling hive desires.
        :returns Dict[str, Worker] selected_workers: a collection of replacements a hive can use w/o guarantees that enough, if at all, replacements are found
        """
        selected_workers: Dict[str, Worker] = {}
        next_round_workers_view: Dict[str, Worker] = {}
        workers_view: Dict[str, Worker] = {key: value for key, value in self.workers if value.status == Status.ONLINE and value.id not in exclusion_dict}

        if not workers_view:
            return selected_workers

        for worker_id, worker in workers_view:
            if worker.uptime > AVG_UPTIME:
                selected_workers[worker_id] = worker
                if len(selected_workers) == quantity:
                    return selected_workers  # desired replacements quantity can only be reached here, if not possible it's mandatory to go to the next for loop
            else:
                next_round_workers_view[worker_id] = worker

        for worker_id, worker in next_round_workers_view:
            selected_workers[worker_id] = worker  # to avoid slowing down the simulation and because a minimum uptime can be defined, we just pick any worker
            if len(selected_workers) == quantity:
                return selected_workers

        return selected_workers

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
