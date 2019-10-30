import os
import json
import math
import logging as log
import numpy as np
import pandas as pd
import domain.metropolis_hastings as mh

from pathlib import Path
from domain.SharedFilePart import SharedFilePart
from domain.Worker import Worker
from domain.Enums import Status, HttpCodes
from domain.helpers.FileData import FileData
from domain.helpers.ConvergenceData import ConvergenceData
from globals.globals import SHARED_ROOT, READ_SIZE, DEFAULT_COLUMN

class Hivemind:
    # region docstrings
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar workers: keeps track workers objects in the simulation
    :type dict<str, domain.Worker>
    :ivar worker_status: keeps track workers objects in the simulation regarding their health
    :type dict<domain.Worker, domain.Status(Enum)>
    :ivar shared_files: part_name is a key to a dict of integer part_id keys leading to actual SharedFileParts
    :type dict<string, dict<int, SharedFilePart>>
    :ivar sf_data: part_name is a key to containing general information about the file
    :type tuple<str, domain.Helpers.FileData>
    :ivar node_uptime_dict: contains each worker node uptime, used to calculate kill probability
    :type dict<str, float>
    :ivar max_stages: number of stages the hive has to converge to the ddv before simulation is considered failed
    :type int
    """
    # endregion

    # region class variables, instance variables and constructors
    __STAGES_WITH_CONVERGENCE = []
    __MAX_CONSECUTIVE_CONVERGENCE_STAGES = 0

    def __init__(self, simfile_name):
        """
        :param simfile_name: path to json file containing the parameters this simulation should execute with
        :type str
        """
        with open(simfile_name) as input_file:
            json_obj = json.load(input_file)
            # Init basic simulation variables
            self.workers = {}
            self.worker_status = {}
            self.shared_files = {}
            self.sf_data = {}
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

    def __set_worker_routing_tables(self, worker, file_name, transition_vector):
        """
        Allows given worker to decide to whom he should send a named file part when he receives it.
        i.e.: Neither workers, nor file parts, have a transition matrix, instead, each worker knows for each named file
        the column vector containing the transition probabilities for that file. For a given file, if all workers were
        merged into one, the concatenation of their column vectors would result into the correct transition matrix.
        :param worker: the worker whose routing table will be updated
        :type domain.Worker
        :param file_name: name of the shared file to which the transition_vector will be mapped to
        :type str
        :param transition_vector: state labeled transition probabilities for all sharers of the named file
        :type list<float>
        """
        worker.set_file_routing(file_name, transition_vector)

    def __uniformely_assign_parts_to_workers(self, shared_files_dict, enforce_online=True):
        """
        Distributes received file parts over the Hive network.
        :param shared_files_dict: receives anyone's dictionary of <file_name, dict<file_number, SharedFilePart>>
        :type dict<str, dict<int, domain.SharedFilePart>>
        :param enforce_online: makes sure receiving workers are online.
        """
        workers_objs = self.__filter_and_map_online_workers() if enforce_online else [*self.worker_status.keys()]
        for file_name, part_number in shared_files_dict.items():
            # Retrive state labels from the file distribution vector
            worker_names = [*self.sf_data[file_name].desired_distribution.index]
            # Quickly filter from the workers which ones are online, fast version of set(a).intersection(b),
            # Do not change positions of file_sharers_names with worker_objs...
            # Doing so changes would make us obtain list<str> containing their names instead of list<domain.Workers>
            choices = [*filter(set(worker_names).__contains__, workers_objs)]
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
            self.__split_shared_file(os.path.join(SHARED_ROOT, file_name))

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
                read_buffer = shared_file.read(READ_SIZE)
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
            self.sf_data[file_name] = FileData(file_name=file_name, parts_count=part_number)
    # endregion

    # region metropolis hastings and transition vector assignment methods
    def __synthesize_transition_matrix(self, adj_matrix, desired_distribution, states):
        """
        :param states: list of worker names who form an hive
        :type list<str>
        :param adj_matrix: adjacency matrix representing connections between various states
        :type list<list<float>>
        :param desired_distribution: column vector representing the distribution that must be achieved by the workers
        :return: A matrix with named lines and columns with the computed transition matrix
        :type pandas.DataFrame
        """
        transition_matrix = mh.metropolis_algorithm(adj_matrix, desired_distribution, column_major_out=True)
        return pd.DataFrame(transition_matrix, index=states, columns=states)

    def __synthesize_shared_files_transition_matrices(self, shared_dict):
        """
        For all keys in the dictionary, obtain file names, the respective adjacency matrix and the desired distribution
        then calculate the transition matrix using metropolis hastings algorithm and feed the result to each worker who
        is a contributor for the survivability of that file
        :param shared_dict: maps file name with extensions to a dictinonary with three keys containing worker_labels who
        are going to receive the file parts associated wih the named file, along with the transition vectors before
        being metropolis hastings processed as well as the desired distributions
        """
        for extended_file_name, markov_chain_data in shared_dict.items():
            file_name = Path(extended_file_name).resolve().stem
            states = markov_chain_data['state_labels']
            adj_matrix = markov_chain_data['adj_matrix']
            desired_distribution = markov_chain_data['ddv']
            # Setting the trackers in this phase speeds up simulation
            self.__set_distribution_trackers(file_name, desired_distribution, states)
            # Compute transition matrix
            transition_matrix = self.__synthesize_transition_matrix(adj_matrix, desired_distribution, states)
            # Split transition matrix into column vectors
            for worker_name in states:
                transition_vector = transition_matrix.loc[:, worker_name]  # <label, value> pairs in column[worker_name]
                self.__set_worker_routing_tables(self.workers[worker_name], file_name, transition_vector)
    # endregion

    # region simulation execution methods
    def execute_simulation(self):
        """
        Runs a stochastic swarm guidance algorithm applied to a P2P network
        """
        online_workers_list = self.__filter_and_map_online_workers()
        for stage in range(self.max_stages):
            self.__try_remove_some_workers(online_workers_list, stage)
            self.__remaining_workers_execute()
            self.__process_stage_results(stage)

    def __try_remove_some_workers(self, online_workers, stage=None):
        """
        For each online worker, if they are online, see if they remain alive for the next stage or if they die,
        according to their uptime record.
        :param online_workers: collection of workers that are known to be online
        :type list<domain.Worker>
        """
        for worker in online_workers:
            uptime = self.node_uptime_dict[worker.name] / 100
            remains_alive = np.random.choice([True, False], p=[uptime, 1 - uptime])
            if not remains_alive:
                self.__remove_worker(worker, clean_kill=False, stage=stage)

    def __rebuild_hive(self, worker):
        # TODO:
        #  future-iterations (simulations don't use orderly leavings and probably never will, thus not urgent):
        #  1. calculate new ddv (uniform distribution of dead node density to other nodes)
        #  2. calculate new transition matrix (feed a new adj matrix to mh algorithm along with new ddv)
        #  3. update FileData fields
        #  4. update any self.sf_* structure as required
        #  5. broadcast new transition.vectors to respective sharers
        #  6. upgrade to byzantine tolerante
        pass

    def __receive_complain(self, suspects_name):
        # TODO:
        #  future-iterations (the goal of the thesis is not to be do a full fledged dependable network, just a demo)
        #  1. When byzantine complaints > threshold
        #       __rebuild_hive(suspect_name)
        pass

    def __remove_worker(self, target, clean_kill=True, stage=None):
        """
        :param target: worker who is going to be removed from the simulation network
        :type domain.Worker
        :param clean_kill: When True worker will ask for his files to be redistributed before leaving the network
        :type bool
        """
        if clean_kill:
            target.leave_hive(orderly=True)
            self.worker_status[target] = Status.OFFLINE
            self.__rebuild_hive(target)
        else:
            target.leave_hive(orderly=False)
            self.worker_status[target] = Status.OFFLINE

            out_file = open("outfile.txt", "a+")

            file_parts = target.request_shared_file_dict()
            for sf_name, sfp_id in file_parts.items():
                data = self.sf_data[sf_name]
                worker_parts_count = len(sfp_id)
                failure_threshold = data.parts_count - math.ceil(data.parts_count * data.highest_density_node_density)

                if worker_parts_count > failure_threshold:
                    worker_names = [*self.sf_data[sf_name].desired_distribution.index]
                    for worker_name in worker_names:
                        worker = self.workers[worker_name]
                        worker.drop_shared_file(shared_file_name=sf_name)
                        self.__drop_shared_file(shared_file_name=sf_name)
                    #   1.1 broadcast message to all workers asking to discard sf_name_parts and sf_transition_vectors
                    out_file.write("Simulation failure at stage {}, for file {}, because of worker {} sudden death...\n"
                                   "worker_parts_count {} > threshold {}, \n"
                                   .format(stage, sf_name, target, worker_parts_count, failure_threshold))
                else:
                    # TODO this-iteration:
                    #  1. assume perfect failure detector
                    #  3. call __rebuild_hive for remaining nodes
                    out_file.write("Simulation recovered from sudden death at stage {}, for file {}...\n"
                                   .format(stage, sf_name, target))
            out_file.close()

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

    def redistribute_files(self, parts):
        """
        :param parts: The parts the caller owned, before announcing his retirement, which will be sent to other workers
        :type dict<str, domain.SharedFilePart>
        """
        self.__uniformely_assign_parts_to_workers(parts, enforce_online=True)
    # endregion

    # region helper methods
    # region setup
    def __set_distribution_trackers(self, file_name, desired_distribution, labels):
        """
        :param file_name: the name of the file to be tracked by the hivemind
        :type str
        :param desired_distribution: the desired distribution vector of the given named file
        :type list<float>
        """
        data = self.sf_data[file_name]
        data.desired_distribution = pd.DataFrame(desired_distribution, index=labels)
        data.current_distribution = pd.DataFrame([0] * len(desired_distribution), index=labels)
        data.convergence_data = ConvergenceData()
    # endregion

    # region stage processing
    def __process_stage_results(self, stage):
        """
        For each file being shared on this hivemind network, check if its desired distribution has been achieved.
        :param stage: stage number - the one that is being processed
        :type int
        """
        if stage == self.max_stages:
            exit(0)

        for data in self.sf_data.values():
            # retrieve from each worker how many parts they have for current data.file_name and update convergence data
            self.__request_file_counts(data)
            # when all queries for a file are done, verify convergence for data.file_name
            self.__check_file_convergence(data, stage)

    def __request_file_counts(self, data):
        worker_names = [*data.desired_distribution.index]
        for worker_name in worker_names:
            worker = self.workers[worker_name]
            if self.worker_status[worker_name] != Status.ONLINE:
                data.current_distribution.at[worker_name, DEFAULT_COLUMN] = 0
            else:
                data.current_distribution.at[worker_name, DEFAULT_COLUMN] = worker.request_file_count(data.file_name)

    def __check_file_convergence(self, data, stage):
        if ConvergenceData.equal_distributions(data.current_distribution, data.desired_distribution):
            data.convergence_data.cswc_increment_and_get(1)
            if data.convergence_data.try_update_convergence_set(stage):
                with open("outfile.txt", "a+") as out_file:
                    out_file.write("File {} converged at stage {}...\n"
                                   "Desired distribution: {},\n"
                                   "Current distribution: {}\n".format(data.file_name,
                                                                       stage,
                                                                       data.desired_distribution.to_string(),
                                                                       data.current_distribution.to_string()
                                                                       )
                                   )
        else:
            data.save_sets_and_reset_data()
    # endregion

    # region teardown
    def __drop_shared_file(self, shared_file_name):
        """
        Hivemind instance stops trackign the named file by removing it from all dictionaries and similar structures
        :param shared_file_name: the name of the file to drop from shared file structures
        """
        try:
            # first try to delete the key while ensuring it exists
            del self.sf_data[shared_file_name]
            del self.shared_files[shared_file_name]
        except KeyError:
            # If error occurs, make sure to clean the key in any dictionary in which it exists after logging the error
            log.error("Key ({}) doesn't exist in at least one shared file tracking structure".format(shared_file_name))
            self.sf_data.pop(shared_file_name, None)
            self.shared_files.pop(shared_file_name, None)
    # endregion

    # region other helpers
    def __filter_and_map_online_workers(self):
        """
        Selects workers (w[0] := worker_status.keys()) who are online (i[1] := worker_status.values())
        :returns Workers objects whose status is online
        :type list<domain.Worker>
        """
        return [*map(lambda w: w[0], [*filter(lambda i: i[1] == Status.ONLINE, self.worker_status.items())])]
    # endregion
    # endregion
