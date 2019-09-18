import json
import numpy as np

from pathlib import Path
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker


class Simulation:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :cvar READ_SIZE: defines the max amount of bytes are read at a time from file to be shared, consequently the parts size
    :type int
    :cvar MY_SHARED_FILES: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar worker_status: keeps track of dead workers in the simulation
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

    def __init__(self, simulation_file_path, shared_file_path):
        """
        :param simulation_file_path: path to json file containing the parameters this simulation should execute with
        :type str
        :param shared_file_path: path to file that this simulation will try persist on the hive network
        :type str
        """
        json_file = json.load(simulation_file_path)

        self.__worker_status = {}
        self.__worker_names = json_file['workers']
        self.__init_workers()
        self.ddv = json_file['ddv']
        self.markov_chain = json_file['transition_vectors']
        self.max_stages = json_file['maxStages']
        self.casualty_chance = json_file['casualtyChance']
        self.multiple_casualties_allowed = json_file['multipleCasualties']

        self.__read_shared_file_bytes(shared_file_path)

    def __init_workers(self):
        for name in self.__worker_names:
            worker = Worker(self, name)
            self.__worker_status[worker] = True

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
                read_buffer = shared_file.read(Simulation.READ_SIZE)
                if read_buffer:
                    part_number = part_number + 1
                    shared_file_part = SharedFilePart(
                        shared_file_name,
                        part_number,
                        read_buffer,
                        self.ddv,
                        (self.__worker_names, self.markov_chain)
                    )
                    shared_file_parts[part_number] = shared_file_part
                else:
                    break
        # TODO review this code
        Simulation.MY_SHARED_FILES[shared_file_name] = shared_file_parts

    def hivemind_send_update(self, worker, shared_file_part):
        if self.worker_status[worker]:
            worker.receive_sfp(shared_file_part)
            return 200
        else:
            return 404

    def execute_simulation(self):
        living = self.__map_and_filter_living_workers
        for i in range(0, self.max_stages):
            if self.casualty_chance > 0.0:
                self.__kill_phase(living)
                living = self.__map_and_filter_living_workers
            for worker in living:
                worker.step()

    def __map_and_filter_living_workers(self):
        return [*map(lambda a_worker: a_worker[0], [*filter(lambda item: item[1], self.worker_status.items())])]

    def __kill_phase(self, living_workers):
        cc = self.casualty_chance
        if self.multiple_casualties_allowed:
            targets = [*filter(lambda dw: np.random.choice([True, False], p=[cc, 1 - cc]), living_workers)]
            for target in targets:
                self.__kill_worker(target)
        else:
            if np.random.choice([True, False], p=[cc, 1 - cc]):
                target = np.random.choice(living_workers)
                self.__kill_worker(target)

    def __kill_worker(self, target):
        # TODO Check if simulation doesn't because this worker is being killed
        target.remove_from_hive(orderly=False)
        self.worker_status[target] = False
