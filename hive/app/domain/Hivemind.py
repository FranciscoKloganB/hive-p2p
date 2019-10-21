import os
import json
import numpy as np
import pandas as pd
import domain.metropolis_hastings as mH

from pathlib import Path
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.Enums import Status, HttpCodes
from domain.helpers.ConvergenceData import ConvergenceData


class Hivemind:
    # region docstrings
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :cvar READ_SIZE: defines the max amount of bytes are read at a time from file to be shared
    :type int
    :ivar worker: keeps track workers objects in the simulation
    :type dict<str, domain.Worker>
    :ivar worker_status: keeps track workers objects in the simulation regarding their health
    :type dict<domain.Worker, domain.Status(Enum)>
    :ivar shared_files: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar sf_desired_distribution: registers the desired distribution for a file, including state labels
    :type dict<str, pandas.Dataframe>
    :ivar sf_current_distribution: keeps track of each shared file distribution, at each discrete time stage
    :type dict<str, list<float>>
    :ivar sf_convergence_data: dictionary that tracks convergence of a given file
    :type dict<str, domain.helpers.ConvergenceData]
    :ivar node_uptime_dict: contains each worker node uptime, used to calculate kill probability
    :type dict<str, float>
    :ivar max_stages: number of stages the hive has to converge to the ddv before simulation is considered failed
    :type int
    """
    # endregion

    # region class variables, instance variables and constructors
    READ_SIZE = 2048
    __SHARED_ROOT = os.path.join(os.getcwd(), 'static', 'shared')
    __STAGES_WITH_CONVERGENCE = []
    __MAX_CONSECUTIVE_CONVERGENCE_STAGES = 0
    __DEFAULT_COLUMN = 0

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
            self.sf_convergence_data = {}
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
    # endregion

    # region domain.Worker related methods
    def __init_workers(self, worker_names):
        """
        Creates worker objects that knows this Hivemind and starts tracking their health
        :param worker_names:
        :type list<str>
        """
        for name in worker_names:
            worker = Worker(self, name)
            self.workers[name] = worker
            self.worker_status[worker] = Status.ONLINE

    def __set_worker_routing_tables(self, worker, file_name, state_labels, transition_vector):
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
        Distributes received file parts over the Hive network.
        :param shared_files_dict: receives anyone's dictionary of <file_name, dict<file_number, SharedFilePart>>
        :type dict<str, dict<int, domain.SharedFilePart>>
        :param enforce_online: makes sure receiving workers are online.
        """
        workers_objs = self.__filter_and_map_online_workers() if enforce_online else [
            *self.worker_status.keys()]
        for file_name, part_number in shared_files_dict.items():
            # Retrive state labels from the file distribution vector
            peer_names = [*self.sf_desired_distribution[file_name].index]
            # Quickly filter from the workers which ones are online, fast version of set(a).intersection(b),
            # Do not change positions of file_sharers_names with worker_objs...
            # Doing so changes would make us obtain list<str> containing their names instead of list<domain.Workers>
            choices = [*filter(set(peer_names).__contains__, workers_objs)]
            for part in part_number.values():
                # Randomly choose a destinatary worker from possible choices and give him the shared file part
                np.random.choice(choices).receive_part(part)
    # endregion

    # region file partitioning methods
    def __split_all_shared_files(self, file_names):
        """
        Obtains the path of all files that are going to be divided for sharing simulation and splits them into parts
        :param file_names: a list containing the name of the files to be shared with their extensions included
        :type str
        """
        for file_name in file_names:
            self.__split_shared_file(os.path.join(self.__SHARED_ROOT, file_name))

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
    # endregion

    # region metropolis hastings and transition vector assignment methods
    def __synthesize_shared_files_transition_matrices(self, shared_dict):
        """
        For all keys in the dictionary, obtain file names, the respective proposal matrix and the desired distribution
        then calculate the transition matrix using metropolis hastings algorithm and feed the result to each worker who
        is a contributor for the survivability of that file
        :param shared_dict: maps file name with extensions to a dictinonary with three keys containing worker_labels who
        are going to receive the file parts associated wih the named file, along with the transition vectors before
        being metropolis hastings processed as well as the desired distributions
        """
        for extended_file_name, markov_chain_data in shared_dict.items():
            file_name = Path(extended_file_name).resolve().stem
            states = markov_chain_data['workers_labels']
            proposal_matrix = markov_chain_data['proposal_matrix']
            desired_distribution = markov_chain_data['ddv']
            # Setting the trackers in this phase speeds up simulation
            self.__set_distribution_trackers(file_name, desired_distribution, states)
            # Compute transition matrix
            transition_matrix = self.__synthesize_transition_matrix(proposal_matrix, desired_distribution, states)
            # Split transition matrix into column vectors
            for worker_name in states:
                transition_vector = [*transition_matrix[worker_name].values]
                self.__set_worker_routing_tables(self.worker_status[worker_name], file_name, states, transition_vector)

    @staticmethod
    def __synthesize_transition_matrix(proposal_matrix, desired_distribution, states):
        """
        :param states: list of worker names who form an hive
        :type list<str>
        :param proposal_matrix: list of probability vectors. Each vector, represents a column, and belogns to the same index label
        :type list<list<float>>
        :param desired_distribution: a single column vector representing the file distribution that must be achieved by the workers
        :return: A matrix with named lines and columns with the computed transition matrix
        :type pandas.DataFrame
        """
        # TODO:
        #  1. metropolis hastings algorithm to synthetize the transition matrix
        #  proposal_matrix = pd.DataFrame(proposal_matrix, index=states, columns=states).transpose() w/o being a df
        #  obtain unlabeled transition_matrix from m-h algorithm, pass as arg the transposed proposal matrix and the ddv
        transition_matrix = mH.metropolis_algorithm(proposal_matrix, desired_distribution)
        return pd.DataFrame(transition_matrix, index=states, columns=states)
    # endregion

    # region simulation execution methods
    def execute_simulation(self):
        """
        Runs a stochastic swarm guidance algorithm applied to a P2P network
        """
        online_workers_list = self.__filter_and_map_online_workers()
        for stage in range(0, self.max_stages):
            self.__try_remove_some_workers(online_workers_list)
            self.__remaining_workers_execute()
            self.__process_stage_results(stage)

    def __try_remove_some_workers(self, online_workers):
        """
        For each online worker, if they are online, see if they remain alive for the next stage or if they die, according
        to their uptime record.
        :param online_workers: collection of workers that are known to be online
        :type list<domain.Worker>
        """
        for worker in online_workers:
            uptime = self.node_uptime_dict[worker.name] / 100
            remains_alive = np.random.choice([True, False], p=[uptime, 1 - uptime])
            if not remains_alive:
                self.__remove_worker(worker, clean_kill=False)

    def __remove_worker(self, target, clean_kill=True):
        """
        :param target: worker who is going to be removed from the simulation network
        :type domain.Worker
        :param clean_kill: When True worker will ask for his files to be redistributed before leaving the network
        :type bool
        """
        if clean_kill:
            target.leave_hive(orderly=True)
            self.worker_status[target] = Status.OFFLINE
        else:
            # TODO:
            #  Check if simulation fails because killed worker had more than N - K parts (see github)
            target.leave_hive(orderly=False)
            self.worker_status[target] = Status.SUSPECT

    def __remaining_workers_execute(self):
        online_workers_list = self.__filter_and_map_online_workers()
        for worker in online_workers_list:
            worker.do_stage()

    def simulate_transmission(self, dest_worker, part):
        """
        :param dest_worker: destinatary of the file part
        :type domain.Worker OR str (domain.Worker.name)
        :param part: the file part to send to specified worker
        :type domain.SharedFilePart
        """
        dest_status = self.worker_status[dest_worker]
        if dest_status == Status.ONLINE:
            dest_worker.receive_part(part)
            return HttpCodes.OK
        elif dest_status == Status.OFFLINE:
            return HttpCodes.SERVER_DOWN
        elif dest_status == Status.SUSPECT:
            return HttpCodes.TIME_OUT
        else:
            return HttpCodes.NOT_FOUND

    def simulate_redistribution(self, parts):
        """
        :param parts: The parts the caller owned, before announcing his retirement, which will be sent to other workers
        :type dict<str, domain.SharedFilePart>
        """
        self.__uniformely_assign_parts_to_workers(parts, enforce_online=True)

    def __process_stage_results(self, stage):
        """
        For each file being shared on this hivemind network, check if its desired distribution has been achieved.
        :param stage: stage number - the one that is being processed
        :type int
        """
        if stage == self.max_stages:
            exit(0)

        for file_name, desired_distribution in self.sf_desired_distribution.items():
            peer_names = [*desired_distribution.index]
            current_distribution = self.sf_current_distribution[file_name]
            # query all workers sharing this file for survivability to tell hivemind how many parts they own from it
            for name in peer_names:
                # Update the stage distribution for this file by querying every worker in its hive
                worker = self.workers[name]
                if self.worker_status[name] != Status.ONLINE:
                    current_distribution.at[name, Hivemind.__DEFAULT_COLUMN] = 0
                else:
                    current_distribution.at[name, Hivemind.__DEFAULT_COLUMN] = worker.request_file_count(file_name)
            # when all queries for a file are done, verify convergence for current file
            data = self.sf_convergence_data[file_name]
            if ConvergenceData.equal_distributions(current_distribution, desired_distribution):
                data.cswc_increment_and_get(1)
                if data.sf_convergence_data[file_name].try_update_convergence_set(stage):
                    print("File: {} converged at stage: {}...".format(file_name, stage))
                    print("Desired Distribution:\n{}".format(desired_distribution.to_string()))
                    print("Current Distribution:\n{}".format(current_distribution.to_string()))
            else:
                data.save_sets_and_reset_data()
    # endregion

    # region helper methods
    def __filter_and_map_online_workers(self):
        """
        :returns Workers objects whose status is online
        :type list<domain.Worker>
        """
        return [*map(
            lambda a_worker: a_worker[0],
            [*filter(lambda item: item[1] == Status.ONLINE, self.worker_status.items())]
        )]

    def __set_distribution_trackers(self, file_name, desired_distribution, labels):
        """
        :param file_name: the name of the file to be tracked by the hivemind
        :type str
        :param desired_distribution: the desired distribution vector of the given named file
        :type list<float>
        """
        self.sf_desired_distribution[file_name] = pd.DataFrame(desired_distribution, index=labels)
        self.sf_current_distribution[file_name] = pd.DataFrame([0] * len(desired_distribution), index=labels)
        self.sf_convergence_data[file_name] = ConvergenceData()
    # endregion
