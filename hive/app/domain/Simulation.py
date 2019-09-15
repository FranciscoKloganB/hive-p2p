import json

from pathlib import Path

from domain.MarkovMatrix import MarkovMatrix
from domain.SharedFilePart import SharedFilePart


class Simulation:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
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
    :ivar shared_file: aggregation of SharedFilePart objects each acting as a container of up to 2KB content blocks
    :type dict<str, SharedFilePart>
    """

    read_size = 2048

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
        self.shared_file = self.__read_shared_file_bytes(shared_file_path)

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
                read_buffer = shared_file.read(Simulation.read_size)
                if read_buffer:
                    part_number = part_number + 1
                    part_id = shared_file_name + str(part_number)
                    with open(part_id, 'w') as out_file:
                        shared_file_part = SharedFilePart(
                            part_id,
                            read_buffer,
                            self.ddv,
                            (self.__workers, self.markov_chain)
                        )
                        shared_file_parts[part_id] = shared_file_part
                        # json.dump(shared_file_part.__dict__, out_file, sort_keys=True, indent=4)
                else:
                    break
        return shared_file_parts

    @staticmethod
    def __read_simulation_file(simulation_file_path):
        """
        :param simulation_file_path: path to a .json file
        :returns a json object based on contents within the pointed file
        """
        return json.load(simulation_file_path)

'''
    def execute(self):
        mm = MarkovMatrix(["A", "B", "C"], [[0.5, 0.5, 0], [0.4, 0.2, 0.4], [0.2, 0.2, 0.6]])
        print(mm.transition_matrix.to_string())
        var = np.random.choice(mm.states, p=mm.transition_matrix["A"])
        print(var)
'''
