from pathlib import Path


class Simulation:
    """Representing a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance"""
    read_size = 2048

    def __init__(self, sm_path, sf_path):
        json_file = __read_simulation_file(sm_path)
        self.worker_count = json_file['workers']
        self.desired_distribution_vector = json_file['ddv']
        self.max_stages = json_file['maxStages']
        self.casualty_chance = json_file['casualtyChance']
        self.multiple_casualties_allowed = json_file['multipleCasualties']
        self.worker_health_status = [True] * worker_count
        self.shared_file = __read_shared_file_bytes(sf_path)
        self.shared_file_parts = {}

    @staticmethod
    def __read_simulation_file(sm_path):
        with open(sm_path, 'r') as json_file:
            return json_file.load(json_file)

    def __read_shared_file_bytes(self, sf_path):
        part_number = 0
        shared_file_name = Path(sf_path).resolve().stem
        with open(sf_path, "rb") as shared_file:
            while True:
                read_buffer = shared_file.read(read_size)
                if read_buffer:
                    part_number = part_number + 1
                    part_id = shared_file_name + str(part_number)
                    with open(part_id, 'w') as sf_part:
                        shared_file_part = SharedFilePart(part_id, read_buffer)
                        self.shared_file_parts[part_id] = shared_file_part
                        json.dump(shared_file_part, sf_part)
                else:
                    break

    def execute(self):
        pass
