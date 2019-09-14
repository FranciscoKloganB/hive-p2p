import json

from pathlib import Path
from domain.SharedFilePart import SharedFilePart


class Simulation:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance

    :param simulation_file_path: points to a json file containing the parameters this simulation should execute with
    :type str
    :param shared_file_path: points to the file that this simulation will try persist on the hive network
    :type str
    """
    read_size = 2048

    def __init__(self, simulation_file_path, shared_file_path):
        json_file = self.__read_simulation_file(simulation_file_path)
        self.worker_count = json_file['workers']
        self.desired_distribution_vector = json_file['ddv']
        self.max_stages = json_file['maxStages']
        self.casualty_chance = json_file['casualtyChance']
        self.multiple_casualties_allowed = json_file['multipleCasualties']
        self.worker_health_status = [True] * self.worker_count
        self.shared_file = self.__read_shared_file_bytes(shared_file_path)

    @staticmethod
    def __read_simulation_file(simulation_file_path):
        """
        :param simulation_file_path: path to a .json file
        :returns a json object based on contents within the pointed file
        """
        return json.load(simulation_file_path)

    @staticmethod
    def __read_shared_file_bytes(shared_file_path):
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
                read_buffer = shared_file.read(Simulation.read_size)
                if read_buffer:
                    part_number = part_number + 1
                    part_id = shared_file_name + str(part_number)
                    with open(part_id, 'w') as out_file:
                        shared_file_part = SharedFilePart(part_id, read_buffer)
                        shared_file_parts[part_id] = shared_file_part
                        json.dump(shared_file_part.__dict__, out_file, sort_keys=True, indent=4)
                else:
                    break
        return shared_file_parts

    def execute(self):
        pass
