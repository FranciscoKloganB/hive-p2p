from __future__ import annotations

import os
import json
import numpy as np

from domain.Hive import Hive
from domain.helpers.Enums import Status
from domain.Worker import Worker
from typing import List, Union, Dict, Any
from domain.helpers.SharedFilePart import SharedFilePart
from globals.globals import SHARED_ROOT, SIMULATION_ROOT, READ_SIZE, MAX_EPOCHS_PLUS


class Hivemind:
    """
    Representation of the P2P Network super node managing one or more hives ---- Simulator is piggybacked
    :ivar int max_epochs: number of stages the hive has to converge to the ddv before simulation is considered disconnected
    :ivar Dict[str, Hive] hives: collection mapping hives' uuid (attribute Hive.id) to the Hive instances
    :ivar Dict[str, Worker] workers: collection mapping workers' names to their Worker instances
    :ivar Dict[str, FileData] files_data: collection mapping file names on the system and their FileData instance
    :ivar Dict[Union[Worker, str], int] workers_status: maps workers or their names to their connectivity status
    :ivar Dict[str, List[FileData]] workers_hives: maps workers' names to hives they are known to belong to
    :ivar Dict[str, float] workers_uptime: maps workers' names to their expected uptime
    :ivar int sim_number: allows different simulation executions to be distinguishable from each other
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, simfile_name: str, sim_number: int) -> None:
        """
        Instantiates an Hivemind object
        :param str simfile_name: path to json file containing the parameters this simulation should execute with
        :param int sim_number: allows different simulation executions to be distinguishable from each other
        """
        self.origin = simfile_name
        self.sim_number = sim_number
        self.epoch = 1

        simfile_path: str = os.path.join(SIMULATION_ROOT, simfile_name)
        with open(simfile_path) as input_file:
            json_obj: Any = json.load(input_file)

            # Init basic simulation variables
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

            shared: Dict[str, Dict[str, Union[List[str], str]]] = json_obj['shared']
            for file_name in shared:
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
        while self.epoch < MAX_EPOCHS_PLUS and self.hives:
            print("epoch: {}".format(self.epoch))
            terminated_hives: List[str] = []
            for hive in self.hives.values():
                hive.execute_epoch(self.epoch)
                if not hive.is_running():
                    terminated_hives.append(hive.id)
                    hive.tear_down(self.origin, self.epoch)
            for hid in terminated_hives:
                print("Hive: {} terminated at epoch {}".format(hid, self.epoch))
                self.hives.pop(hid)
            self.epoch += 1
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

    # region Peer Search and Cloud References
    def find_replacement_worker(self, exclusion_dict: Dict[str, Worker], quantity: int) -> Dict[str, Worker]:
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param Dict[str, Worker] exclusion_dict: collection of worker ids that the calling hive haves no interest in, for any reason
        :param int quantity: how many replacements the calling hive desires.
        :returns Dict[str, Worker] selected_workers: a collection of replacements a hive can use w/o guarantees that enough, if at all, replacements are found
        """
        selected_workers: Dict[str, Worker] = {}
        workers_view = self.workers.copy().values()
        for worker in workers_view:
            if len(selected_workers) == quantity:
                return selected_workers
            elif worker.status != Status.ONLINE:
                self.workers.pop(worker.id, None)
            elif worker.id not in exclusion_dict:
                selected_workers[worker.id] = worker
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
        initial_members: np.array = np.random.choice(a=[*self.workers.keys()], size=shared[file_name]['hive_size'], replace=False)
        for member_id in initial_members:
            hive_members[member_id] = self.workers[member_id]
        hive = Hive(self, file_name, hive_members, sim_number=self.sim_number, origin=self.origin)
        self.hives[hive.id] = hive
        return hive
    # endregion
