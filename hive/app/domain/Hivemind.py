import os
import json
import numpy as np
import pandas as pd

from pathlib import Path
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.Enums import Status, HttpCodes


class Hivemind:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :cvar READ_SIZE: defines the max amount of bytes are read at a time from file to be shared
    :type int
    :ivar shared_files: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar worker_status: keeps track workers in the simulation
    :type dict<str, bool>
    :ivar max_stages: number of stages the hive has to converge to the ddv before simulation is considered failed
    :type int
    :ivar node_uptime_dict: contains each worker node uptime, used to calculate kill probability
    :type dict<str, float>
    :ivar workers: A list of all workers known to the Hivemind
    :type list<domain.Worker>
    :ivar __ddv: stochastic like list to define the desired distribution vector the hive should reach before max_stages
    :type list<float>
    :ivar __markov_columns: list containing lists, each defining jump probabilities of each state between stages
    :type list<list<float>>
    """

    READ_SIZE = 2048
    SHARED_ROOT = os.path.join(os.getcwd(), 'static', 'shared')
    STAGES_WITH_CONVERGENCE = []
    MAX_CONSECUTIVE_CONVERGENCE_STAGES = 0

    def __init__(self, simfile_path):
        """
        :param simfile_path: path to json file containing the parameters this simulation should execute with
        :type str
        """
        with open(simfile_path) as json_obj:
            json_obj = json.load(simfile_path)
            # Init basic simulation variables
            self.shared_files = {}
            self.worker_status = {}
            self.max_stages = json_obj['max_simulation_stages']
            self.node_uptime_dict = json_obj['nodes_uptime']
            self.workers = [*self.node_uptime_dict.keys()]
            # Create the P2P network nodes (domain.Workers) without any job
            self.__init_workers()
            # Read and split all shareable files specified on the input
            self.__split_all_shared_files([*json_obj['shared'].keys()])
            # For all shareable files, set that shared file_routing table
            self.__set_worker_routing_tables(json_obj['shared'])
            self.__ddv = json_obj['ddv']
            self.__markov_columns = json_obj['transition_vectors']
            # Distribute files before starting simulation
            self.__uniformely_assign_parts_to_workers(self.shared_files, enforce_online=False)

    def __init_workers(self):
        worker_names = self.workers
        worker_count = len(worker_names)
        for i in range(0, worker_count):
            # Create a named worker that knows his Super Node (Hivemind) and list him as online on the hivemind
            self.workers[i] = Worker(self, worker_names[i])
            self.worker_status[self.workers[i]] = Status.ONLINE

    def __split_all_shared_files(self, file_names):
        """
        Obtains the path of all files that are going to be divided for sharing simulation and splits them into parts
        :param file_names: a list containing the name of the files to be shared with their extensions included
        :type str
        """
        for file_name in file_names:
            self.__split_shared_file(os.path.join(self.SHARED_ROOT, file_name))

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
                read_buffer = shared_file.read(Hivemind.READ_SIZE)
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

    def __set_worker_routing_tables(self, shared_dict):
        """
        :param shared_dict: maps file name with extensions to a dictinonary with three keys containing worker_labels who
        are going to receive the file parts associated wih the named file, along with the transition vectors before being
        metropolis hastings processed as well as the desired distributions # TODO make it proper
        :type dict<str, dict<str, obj>
        """
        for extended_name, markov_chain_data in shared_dict.items():
            file_name = Path(extended_name).resolve().stem
            df = pd.DataFrame(transition_probabilities, index=markov_chain_data['workers_labels'])
            for worker in self.workers:
                df.columns = list(worker.name)
                worker.set_file_routing(file_name, df)

    def __uniformely_assign_parts_to_workers(self, shared_files_dict, enforce_online=True):
        # iterate dict<part_name, dict<part_number, shared file part object>>
        for name, parts in shared_files_dict.items():
            for part in parts.values:
                # choose a worker to receive this part
                worker_obj = np.random.choice(self.__filter_and_map_online_workers() if enforce_online else self.workers)
                worker_obj.receive_part(part, no_check=True)

    def __filter_and_map_online_workers(self):
        """
        Filters all worker items(key:val) from the worker_status dict known to this hivemind who have Status.Online
        :returns list<domain.Worker>: a list containing only the keys (workers, w/o their status) returned by the filter
        """
        filtered_items = [*filter(lambda item: item[1] == Status.ONLINE, self.worker_status.items())]
        return [*map(lambda a_worker: a_worker[0], filtered_items)]

    def __kill_phase(self, workers):
        """
        :param workers: collection of workers that are known to be online
        :type list<domain.Worker>
        """
        cc = self.casualty_chance
        if self.multiple_casualties_allowed:
            targets = [*filter(lambda dw: np.random.choice([True, False], p=[cc, 1 - cc]), workers)]
            for target in targets:
                self.__kill_worker(target, clean_kill=False)
        else:
            if np.random.choice([True, False], p=[cc, 1 - cc]):
                target = np.random.choice(workers)
                self.__kill_worker(target, clean_kill=False)

    def __kill_worker(self, target, clean_kill=True):
        """
        :param target: worker who is going to be removed from the simulation network
        :type domain.Worker
        :param clean_kill: When True worker will ask for his files to be redistributed before leaving the network
        :type bool
        """
        if clean_kill:
            target.leave_hive(orderly=True)
        else:
            # TODO Check if simulation fails because killed worker had more than N - K parts (see github)
            target.leave_hive(orderly=False)
        self.worker_status[target] = Status.SUSPECT

    def __process_stage_results(self, shared_file_name, stage, cswc_count):
        # TODO Create a margin of error that defines stage_distribution == self.ddv equality
        stage_distribution = []
        for worker in self.workers:
            stage_distribution.append(worker.request_file_count(shared_file_name))
        if stage_distribution == self.__ddv:
            cswc_count += 1
            if cswc_count > Hivemind.MAX_CONSECUTIVE_CONVERGENCE_STAGES:
                Hivemind.MAX_CONSECUTIVE_CONVERGENCE_STAGES = cswc_count
            Hivemind.STAGES_WITH_CONVERGENCE.append(stage)
            return cswc_count
        else:
            return - cswc_count

    def execute_simulation(self):
        consecutive_convergences = 0
        workers = self.__filter_and_map_online_workers()
        for stage in range(0, self.max_stages):
            if self.casualty_chance > 0.0:
                self.__kill_phase(workers)
                workers = self.__filter_and_map_online_workers()
            for worker in workers:
                worker.do_stage()
            consecutive_convergences += self.__process_stage_results(
                [*(self.shared_files.keys())][0], stage, consecutive_convergences
            )

    def simulate_transmission(self, worker, part):
        """
        :param worker: destination of the file part
        :type domain.Worker
        :param part: the file part to send to specified worker
        :type domain.SharedFilePart
        """
        if self.worker_status[worker] == Status.ONLINE:
            worker.receive_part(part)
            return HttpCodes.OK
        elif not self.worker_status[worker] == Status.OFFLINE:
            return HttpCodes.NOT_FOUND
        else:
            return HttpCodes.TIME_OUT

    def simulate_redistribution(self, parts):
        """
        :param parts: The parts the caller owned, before announcing his retirement, which will be sent to other workers
        :type dict<str, domain.SharedFilePart>
        """
        self.__uniformely_assign_parts_to_workers(parts, enforce_online=True)
