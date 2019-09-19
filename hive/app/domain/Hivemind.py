import json
import numpy as np

from pathlib import Path
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.Enums import Status, HttpCodes


class Hivemind:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :cvar READ_SIZE: defines the max amount of bytes are read at a time from file to be shared
    :type int
    :cvar MY_SHARED_FILES: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar worker_status: keeps track workers in the simulation
    :type dict<str, bool>
    :ivar worker_names: simple list of names to avoid repetitive unpacking of worker_status.keys()
    :type list<str>
    :ivar ddv: stochastic like list to define the desired distribution vector the hive should reach before max_stages
    :type list<float>
    :ivar markov_chain: list containing lists, each defining jump probabilities of each state between stages
    :type list<list<float>>
    :ivar max_stages: number of stages the hive has to converge to the ddv before simulation is considered failed
    :type int
    :ivar casualty_chance: probability of having one random worker leaving the hive per stage.
    :type float
    :ivar multiple_casualties_allowed: defines the possibility of more than a worker leaving the hive at each stage.
    :type bool: when True casualty_chance is calculated independently for each worker!
    """

    READ_SIZE = 2048
    MY_SHARED_FILES = {}
    STAGES_WITH_CONVERGENCE = []
    MAX_CONSECUTIVE_CONVERGENCE_STAGES = 0

    def __init__(self, simulation_file_path, shared_file_path):
        """
        :param simulation_file_path: path to json file containing the parameters this simulation should execute with
        :type str
        :param shared_file_path: path to file that this simulation will try persist on the hive network
        :type str
        """
        json_file = json.load(simulation_file_path)
        self.__worker_status = {}
        self.__workers = json_file['workers']
        self.__init_workers()
        self.ddv = json_file['ddv']
        self.markov_chain = json_file['transition_vectors']
        self.max_stages = json_file['maxStages']
        self.casualty_chance = json_file['casualtyChance']
        self.multiple_casualties_allowed = json_file['multipleCasualties']
        self.__read_shared_file_bytes(shared_file_path)

    def __init_workers(self):
        worker_count = len(self.__workers)
        for name in range(0, worker_count):
            self.__workers[name] = Worker(self, name)
            self.__worker_status[self.__workers[name]] = Status.ONLINE

    def __read_shared_file_bytes(self, shared_file_path):
        """
        Reads a file from disk which the simulation wants to persist on the hive network.
        The contents of the file are read in 2KB blocks and are encapsulated along with their ID and SHA256 for proper
        distribution on the hive.
        :param shared_file_path: path to an arbitrary file to persist on the hive network
        :returns the raw content of the file, used to assert if simulation was successful after max_stages happens
        """
        shared_file_parts = {}
        shared_file_name = Path(shared_file_path).resolve().stem
        with open(shared_file_path, "rb") as shared_file:
            part_number = 0
            while True:
                read_buffer = shared_file.read(Hivemind.READ_SIZE)
                if read_buffer:
                    part_number = part_number + 1
                    shared_file_part = SharedFilePart(
                        shared_file_name,
                        part_number,
                        read_buffer,
                        self.ddv,
                        (self.__workers, self.markov_chain)
                    )
                    shared_file_parts[part_number] = shared_file_part
                else:
                    break
        Hivemind.MY_SHARED_FILES[shared_file_name] = shared_file_parts

    def __filter_and_map_living_workers(self):
        filtered_items = [*filter(lambda item: item[1] == Status.ONLINE, self.worker_status.items())]
        return [*map(lambda a_worker: a_worker[0], filtered_items)]

    def __kill_phase(self, living_workers):
        cc = self.casualty_chance
        if self.multiple_casualties_allowed:
            targets = [*filter(lambda dw: np.random.choice([True, False], p=[cc, 1 - cc]), living_workers)]
            for target in targets:
                self.__kill_worker(target, clean_kill=False)
        else:
            if np.random.choice([True, False], p=[cc, 1 - cc]):
                target = np.random.choice(living_workers)
                self.__kill_worker(target, clean_kill=False)

    def __kill_worker(self, target, clean_kill=True):
        if clean_kill:
            target.leave_hive(orderly=True)
        else:
            # TODO Check if simulation fails because killed worker had more than N - K parts (see github)
            target.leave_hive(orderly=False)
        self.worker_status[target] = Status.SUSPECT

    def __process_stage_results(self, shared_file_name, stage, cswc_count):
        # TODO Create a margin of error that defines stage_distribution == self.ddv equality
        stage_distribution = []
        for worker in self.__workers:
            stage_distribution.append(worker.request_file_count(shared_file_name))
        if stage_distribution == self.ddv:
            cswc_count += 1
            if cswc_count > Hivemind.MAX_CONSECUTIVE_CONVERGENCE_STAGES:
                Hivemind.MAX_CONSECUTIVE_CONVERGENCE_STAGES = cswc_count
            Hivemind.STAGES_WITH_CONVERGENCE.append(stage)
            return cswc_count
        else:
            return - cswc_count

    def execute_simulation(self):
        consecutive_convergences = 0
        living = self.__filter_and_map_living_workers()
        for stage in range(0, self.max_stages):
            if self.casualty_chance > 0.0:
                self.__kill_phase(living)
                living = self.__filter_and_map_living_workers()
            for worker in living:
                worker.do_stage()
            consecutive_convergences += self.__process_stage_results(
                [*(Hivemind.MY_SHARED_FILES.keys())][0], stage, consecutive_convergences
            )

    def simulate_transmission(self, worker, part):
        if self.worker_status[worker] == Status.ONLINE:
            worker.receive_part(part)
            return HttpCodes.OK
        elif not self.worker_status[worker] == Status.OFFLINE:
            return HttpCodes.NOT_FOUND
        else:
            return HttpCodes.TIME_OUT

    def simulate_redistribution(self, parts):
        online_workers = self.__filter_and_map_living_workers()
        for part in parts.values():
            dest_worker = np.random.choice(online_workers)
            dest_worker.receive_part(part)
