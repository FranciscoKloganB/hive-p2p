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
    :ivar workers: maps worker_name (key) to worker obj (value)
    :type dict<string, domain.Worker>
    :ivar worker_status: keeps track workers in the simulation
    :type dict<str, ENUM>
    :ivar shared_files: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar sf_desired_distribution: registers the desired destribution of a shared file, file_name is key
    :type dict<str, list<float>>
    :ivar sf_current_distribution: keeps track of each shared file distribution, at each discrete time stage
    :type dict<str, list<float>>
    :ivar node_uptime_dict: contains each worker node uptime, used to calculate kill probability
    :type dict<str, float>
    :ivar max_stages: number of stages the hive has to converge to the ddv before simulation is considered failed
    :type int
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
            self.workers = {}
            self.worker_status = {}
            self.shared_files = {}
            self.sf_desired_distribution = {}
            self.sf_current_distribution = {}
            self.node_uptime_dict = json_obj['nodes_uptime']
            self.max_stages = json_obj['max_simulation_stages']
            # Create the P2P network nodes (domain.Workers) without any job
            self.__init_workers([*self.node_uptime_dict.keys()])
            # Read and split all shareable files specified on the input
            self.__split_all_shared_files([*json_obj['shared'].keys()])
            # For all shareable files, set that shared file_routing table
            self.__synthesize_shared_files_transition_matrices(json_obj['shared'])
            # Distribute files before starting simulation
            self.__uniformely_assign_parts_to_workers(self.shared_files, enforce_online=False)

    def __init_workers(self, worker_names):
        """
        Creates worker objects that knows this Hivemind and starts tracking their health
        :param worker_names:
        :type list<str>
        """
        for name in worker_names:
            self.workers[name] = Worker(self, name)
            self.worker_status[name] = Status.ONLINE

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

    def __synthesize_shared_files_transition_matrices(self, shared_dict):
        """
        For all keys in the dictionary, obtain file names, the respective proposal matrix and the desired distribution
        then calculate the transition matrix using metropolis hastings algorithm and feed the result to each worker who
        is a contributor for the survivability of that file
        :param shared_dict: maps file name with extensions to a dictinonary with three keys containing worker_labels who
        are going to receive the file parts associated wih the named file, along with the transition vectors before being
        metropolis hastings processed as well as the desired distributions
        """
        for extended_file_name, markov_chain_data in shared_dict.items():
            file_name = Path(extended_file_name).resolve().stem
            state_labels = markov_chain_data['workers_labels']
            proposal_matrix = markov_chain_data['proposal_matrix']
            desired_distribution = markov_chain_data['ddv']
            # Setting the trackers in this phase speeds up simulation
            self.__set_distribution_trackers(file_name, desired_distribution)
            # Compute transition matrix
            transition_matrix = self.__synthesize_transition_matrix(state_labels, proposal_matrix, desired_distribution)
            # Split transition matrix into column vectors
            for worker_name in state_labels:
                transition_vector = [*transition_matrix[worker_name].values]
                self.__set_worker_routing_tables(self.workers[worker_name], file_name, state_labels, transition_vector)

    def __set_distribution_trackers(self, file_name, desired_distribution):
        """
        :param file_name: the name of the file to be tracked by the hivemind
        :type str
        :param desired_distribution: the desired distribution vector of the given named file
        :type list<float>
        """
        self.sf_desired_distribution[file_name] = desired_distribution
        self.sf_current_distribution[file_name] = [0] * len(desired_distribution)

    @staticmethod
    def __synthesize_transition_matrix(state_labels, proposal_matrix, desired_distribution):
        """
        TODO:
            1. metropolis hastings algorithm to synthetize the transition matrix
         Reminder don't forget to transpose the input vectors the dataframe might end up being represented as line vector
         transitions instead of column vector transitions as desired! This transposition isn't necessary with 1-D arrays
         but is when we are passing matrices (N-D Arrays)
        :param state_labels: list of worker names who form an hive
        :type list<str>
        :param proposal_matrix: list of probability vectors. Each vector, represents a column, and belogns to the same index label
        :type list<list<float>>
        :param desired_distribution: a single column vector representing the file distribution that must be achieved by the workers
        :return: A matrix with named lines and columns with the computed transition matrix
        :type pandas.DataFrame
        """
        transition_matrix = None
        return pd.DataFrame(transition_matrix, index=state_labels, columns=state_labels)

    @staticmethod
    def __set_worker_routing_tables(worker, file_name, state_labels, transition_vector):
        """
        Allows given worker to decide to whom he should send a named file part when he receives it.
        i.e.: Neither workers, nor file parts, have a transition matrix, instead, each worker knows for each named file
        the column vector containing the transition probabilities for that file. For a given file, if all workers were
        merged into one, the concatenation of their column vectors would result into the correct transition matrix.
        """
        df = pd.DataFrame(transition_vector, index=state_labels, columns=[*worker.name])
        worker.set_file_routing(file_name, df)

    def __uniformely_assign_parts_to_workers(self, shared_files_dict, enforce_online=True):
        """
        TODO:
            Only some nodes responsible for a given file should be selected regardless of their Connection Status
        Distributes received file parts over the Hive network.
        :param shared_files_dict: receives anyone's dictionary of <file_name, dict<file_number, SharedFilePart>>
        :type dict<str, dict<int, domain.SharedFilePart>>
        :param enforce_online: makes sure receiving workers are online.
        """
        for name, parts in shared_files_dict.items():
            for part in parts.values:
                # choose a worker to receive this part
                worker_obj = np.random.choice(self.__filter_and_map_online_workers() if enforce_online else self.workers)
                worker_obj.receive_part(part, no_check=True)

    def __filter_and_map_online_workers(self):
        """
        Filters all worker items(worker_name:Status) from the worker_status dict known to this hivemind who have Status.Online
        :returns a list containing only the keys (workers names, w/o their status) returned by the first filter
        :type list<str>
        """
        filtered_workers = [*filter(lambda item: item[1] == Status.ONLINE, self.worker_status.items())]
        return [*map(lambda a_worker_name: a_worker_name[0], filtered_workers)]

    def __remove_workers_phase(self, online_workers):
        """
        For each online worker, if they are online, see if they remain alive for the next stage or if they die, according
        to their uptime record.
        :param online_workers: collection of workers that are known to be online
        :type list<domain.Worker>
        """
        for name in online_workers:
            uptime = self.node_uptime_dict[name] / 100
            remains_alive = np.random.choice([True, False], p=[uptime, 1 - uptime])
            if not remains_alive:
                self.__remove_worker(name, clean_kill=False)

    def __remove_worker(self, target, clean_kill=True):
        """
        :param target: name of the worker who is going to be removed from the simulation network
        :type str
        :param clean_kill: When True worker will ask for his files to be redistributed before leaving the network
        :type bool
        """
        if clean_kill:
            self.workers[target].leave_hive(orderly=True)
            self.worker_status[target] = Status.OFFLINE
        else:
            # TODO Check if simulation fails because killed worker had more than N - K parts (see github & ask dani for algorithm)
            self.workers[target].leave_hive(orderly=False)
            self.worker_status[target] = Status.SUSPECT

    def __process_stage_results(self, stage, cswc_count):
        """
        TODO
            Redo this! There must be an effecient way to do this.
        For each file being shared on this hivemind network, check if its desired distribution has been, trully, achieved.
        :param stage: stage number - the one that is being processed
        :type int
        :param cswc_count: value, for the named file, that counts for how many consecutive steps has the ddv been respected
        :type int
        """
        # For each file name the hivemind is controlling, create a vector
        for file_name in self.sf_desired_distribution.keys():
            for worker in self.workers.values():
                stage_distribution.append(worker.request_file_count(file_name))
                if self.sf_current_distribution[file_name] == self.sf_desired_distribution[file_name]:
                    cswc_count += 1
                    if cswc_count > Hivemind.MAX_CONSECUTIVE_CONVERGENCE_STAGES:
                        Hivemind.MAX_CONSECUTIVE_CONVERGENCE_STAGES = cswc_count
                    Hivemind.STAGES_WITH_CONVERGENCE.append(stage)
                    return cswc_count
                else:
                    return - cswc_count

    def execute_simulation(self):
        """
        TODO:
            Modify __filter_and_map_online_workers to work with dictionaries or modify this function to convert the list properly
        Runs a stochastic swarm guidance algorithm applied to a P2P network
        """
        cswc_count = 0
        workers = self.__filter_and_map_online_workers()
        for stage in range(0, self.max_stages):
            self.__remove_workers_phase(workers)
            workers = self.__filter_and_map_online_workers()
            for worker in workers:
                worker.do_stage()
            # TODO:
            #   Correct the line below to work for multiple files.
            cswc_count += self.__process_stage_results([*(self.shared_files.keys())][0], stage, cswc_count)

    def simulate_transmission(self, worker, part):
        """
        TODO:
            Either convert the worker_status to be a dict of domain.Worker : Status, like it apparently was before
            Or
            Modify this to be work with names instead of Objects!
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
