import json
import numpy as np

from pathlib import Path
from domain.SharedFilePart import SharedFilePart


class Simulation:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :cvar READ_SIZE: defines the max amount of bytes are read at a time from file to be shared, consequently the parts size
    :type int
    :cvar MY_SHARED_FILES: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar ddv: stochastic like list to define the desired distribution vector the hive should reach before max_stages
    :type list<float>
    :ivar markov_chain: list containing lists, each defining jump probabilities of each state between stages
    :type list<list<float>>
    :ivar worker_status: keeps track of dead workers in the simulation
    :type dict<str, bool>
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
        json_file = self.__read_simulation_file(simulation_file_path)
        self.__workers = json_file['workers']
        self.ddv = json_file['ddv']
        self.markov_chain = json_file['transition_vectors']
        self.worker_status = dict.fromkeys(self.__workers, True)
        self.max_stages = json_file['maxStages']
        self.casualty_chance = json_file['casualtyChance']
        self.multiple_casualties_allowed = json_file['multipleCasualties']
        self.__read_shared_file_bytes(shared_file_path)

    @staticmethod
    def __read_simulation_file(simulation_file_path):
        """
        :param simulation_file_path: path to a .json file
        :returns a json object based on contents within the pointed file
        """
        return json.load(simulation_file_path)

    def __read_shared_file_bytes(self, shared_file_path):
        """
        Reads a file from disk which the simulation wants to persist on the hive network.
        The contents of the file are read in 2KB blocks and are encapsulated along with their ID and SHA256 for proper
        distribution on the hive.
        :param shared_file_path: path to an arbitrary file to persist on the hive network
        :returns the raw content of the file, used to assert if simulation was successful after max_stages happens
        """
        part_number = 0
        shared_file_parts = {}
        shared_file_name = Path(shared_file_path).resolve().stem
        with open(shared_file_path, "rb") as shared_file:
            while True:
                read_buffer = shared_file.read(Simulation.READ_SIZE)
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
        Simulation.MY_SHARED_FILES[shared_file_name] = shared_file_parts

    def hivemind_send_update(self, worker, shared_file_part):
        if self.worker_status[worker]:
            worker.receive_sfp(shared_file_part)
            return 200
        else:
            return 404

    def __run_stage(self):
        for worker in self.__workers:
            worker.send_sfp()

    def __kill_phase(self):
        cc = self.casualty_chance
        if cc > 0.0:
            living = [*map(lambda k: k[0], [*filter(lambda i: i[1], self.worker_status.items())])]
            if not self.multiple_casualties_allowed:
                if np.random.choice([True, False], p=[cc, 1 - cc]):
                    dead_worker = np.random.choice(living)
                    self.__kill_worker(dead_worker)
            else:
                dead_workers = [*filter(lambda dw: np.random.choice([True, False], p=[cc, 1 - cc]), living)]
                for worker in dead_workers:
                    self.__kill_worker(worker)

    def __kill_worker(self, worker):
        worker.remove_from_hive(orderly=False)
        self.worker_status[worker] = False
