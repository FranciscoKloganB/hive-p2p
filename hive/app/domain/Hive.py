from __future__ import annotations

import math
import uuid
import traceback
from operator import itemgetter

import numpy as np
import pandas as pd
import domain.Hivemind as hm
import utils.matrices as matrices
import utils.transition_matrices as tmg

from domain.Worker import Worker
from typing import Dict, List, Any, Tuple
from domain.helpers.FileData import FileData
from domain.helpers.Enums import Status, HttpCodes
from domain.helpers.SharedFilePart import SharedFilePart
from globals.globals import REPLICATION_LEVEL, DEFAULT_COL, TRUE_FALSE, COMMUNICATION_CHANCES, MAX_EPOCHS


class Hive:
    """
    :ivar int current_epoch: tracks the epoch at which the Hive is currently at
    :ivar List[float, float] corruption_chances: used to simulate file corruption on behalf of the workers, to avoid keeping independant distributions for each part and each replica
    :ivar str id: unique identifier in str format
    :ivar Hivemind hivemind: reference to the master server, which in this case is just a simulator program
    :ivar Dict[str, Worker] members: Workers that belong to this P2P Hive, key is worker.id, value is the respective Worker instance
    :ivar FileData file: instance of class FileData which contains information regarding the file persisted by this hive
    :ivar DataFrame desired_distribution: distribution hive members are seeking to achieve for each the files they persist together.
    :ivar int critical_size: minimum number of replicas required for data recovery plus the number of peer faults the system must support during replication.
    :ivar int sufficient_size: depends on churn-rate and equals critical_size plus the number of peers expected to fail between two successive recovery phases
    :ivar int original_size: stores the initial hive size
    :ivar int redundant_size: application-specific system parameter, but basically represents that the hive is to big
    :ivar int set_recovery_epoch_sum: stores the sum of the values returned by all SharedFilePart.set_recovery_epoch calls - used for simulation output purposes
    :ivar int set_recovery_epoch_calls: stores how many times SharedFilePart.set_recovery_epoch calls was called  during the current epoch
    :ivar bool running: indicates if the hive has terminated - used for simulation purposes
    """

    # region Class Variables, Instance Variables and Constructors

    def __init__(self, hivemind: hm.Hivemind, file_name: str, members: Dict[str, Worker], sim_number: int = 0, origin: str = "") -> None:
        """
        Instantiates an Hive abstraction
        :param Hivemind hivemind: Hivemand instance object which leads the simulation
        :param str file_name: name of the file this Hive is responsible for
        :param Dict[str, Worker] members: collection mapping names of the Hive's initial workers' to their Worker instances
        :param int sim_number: optional value that can be passed to FileData to generate different .out names
        """
        self.current_epoch: int = 0
        self.corruption_chances: List[float] = [0, 0]
        self.id: str = str(uuid.uuid4())
        self.hivemind = hivemind
        self.members: Dict[str, Worker] = members
        self.file: FileData = FileData(file_name, sim_number=sim_number, origin=origin)
        self.desired_distribution: pd.DataFrame = pd.DataFrame()
        self.critical_size: int = REPLICATION_LEVEL
        self.sufficient_size: int = self.critical_size + math.ceil(len(self.members) * 0.34)
        self.original_size: int = len(members)
        self.redundant_size: int = self.sufficient_size + len(self.members)
        self.running: bool = True
        self.set_recovery_epoch_sum: int = 0
        self.set_recovery_epoch_calls: int = 0
        self.create_and_bcast_new_transition_matrix()

    # endregion

    # region Routing

    def remove_cloud_reference(self) -> None:
        """
        TODO: future-iterations
        Remove cloud references and delete files within it
        """
        pass

    def add_cloud_reference(self) -> None:
        """
        TODO: future-iterations
        Remaining hive members upload all data they have to a cloud server
        """
        # noinspection PyUnusedLocal
        cloud_ref: str = self.hivemind.get_cloud_reference()

    def route_part(self, sender: str, destination: str, part: SharedFilePart, fresh_replica: bool = False) -> Any:
        """
        Receives a shared file part and sends it to the given destination
        :param str sender: id of the worker sending the message
        :param str destination: destination worker's id
        :param SharedFilePart part: the file part to send to specified worker
        :param bool fresh_replica: stops recently created replicas from being corrupted, since they are not likely to be corrupted in disk
        :returns int: http codes based status of destination worker
        """
        if sender == destination:
            return HttpCodes.DUMMY

        self.file.simulation_data.set_moved_parts_at_index(1, self.current_epoch)

        if np.random.choice(a=TRUE_FALSE, p=COMMUNICATION_CHANCES):
            self.file.simulation_data.set_lost_messages_at_index(1, self.current_epoch)
            return HttpCodes.TIME_OUT

        if not fresh_replica and np.random.choice(a=TRUE_FALSE, p=self.corruption_chances):
            self.file.simulation_data.set_corrupt_files_at_index(1, self.current_epoch)
            return HttpCodes.BAD_REQUEST

        member: Worker = self.members[destination]
        if member.status == Status.ONLINE:
            return member.receive_part(part)
        else:
            return HttpCodes.NOT_FOUND

    # endregion

    # region Swarm Guidance

    def new_desired_distribution(self, member_ids: List[str], member_uptimes: List[float]) -> List[float]:
        """
        Normalizes inputted member uptimes and saves it on Hive.desired_distribution attribute
        :param List[str] member_ids: list of member ids representing the current hive membership
        :param List[float] member_uptimes: list of member uptimes to be normalized
        :returns List[float] desired_distribution: uptimes represent 'reliability', thus, desired distribution is the normalization of the members' uptimes
        """
        uptime_sum = sum(member_uptimes)
        uptimes_normalized = [member_uptime / uptime_sum for member_uptime in member_uptimes]

        self.desired_distribution = pd.DataFrame(data=uptimes_normalized, index=member_ids)
        self.file.new_desired_distribution(self.desired_distribution, member_ids)

        return uptimes_normalized

    def new_transition_matrix(self) -> pd.DataFrame:
        """
        returns DataFrame: Creates a new transition matrix for the members of the Hive, to be followed independently by each of them
        """
        member_uptimes: List[float] = []
        member_ids: List[str] = []

        for worker in self.members.values():
            member_uptimes.append(worker.uptime)
            member_ids.append(worker.id)

        A: np.ndarray = np.asarray(matrices.new_symmetric_adjency_matrix(len(member_ids)))
        v_: np.ndarray = np.asarray(self.new_desired_distribution(member_ids, member_uptimes))

        T = self.select_fastest_topology(A, v_)

        return pd.DataFrame(T, index=member_ids, columns=member_ids)

    def broadcast_transition_matrix(self, transition_matrix: pd.DataFrame) -> None:
        """
        Gives each member his respective slice (vector column) of the transition matrix the Hive is currently executing.
        post-scriptum: we could make an optimization that sets a transition matrix for the hive, ignoring the file names, instead of mapping different file
        names to an equal transition matrix within each hive member, thus reducing space overhead arbitrarly, however, this would make Simulation harder. This
        note is kept for future reference. This also assumes an hive can store multiple files. For simplicity each Hive only manages one file for now.
        """
        for worker in self.members.values():
            transition_vector: pd.DataFrame = transition_matrix.loc[:, worker.id]
            worker.set_file_routing(self.file.name, transition_vector)

    # endregion

    # region Simulation Interface

    # noinspection DuplicatedCode
    def spread_files(self, spread_mode: str, file_parts: Dict[int, SharedFilePart]):
        """
        Spreads files over the initial members of the Hive
        :param str spread_mode: 'u' for uniform distribution, 'a' one* peer receives all or 'i' to distribute according to the desired steady state distribution
        :param Dict[int, SharedFilePart] file_parts: file parts to distribute over the members
        """
        self.file.simulation_data.initial_spread = spread_mode

        if spread_mode == "a":
            choices: List[Worker] = [*self.members.values()]
            workers: List[Worker] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
            for worker in workers:
                for part in file_parts.values():
                    part.references += 1
                    worker.receive_part(part)

        elif spread_mode == "u":
            for part in file_parts.values():
                choices: List[Worker] = [*self.members.values()]
                workers: List[Worker] = np.random.choice(a=choices, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    part.references += 1
                    worker.receive_part(part)

        elif spread_mode == 'i':
            choices = [*self.members.values()]
            desired_distribution: List[float] = []
            for member_id in choices:
                desired_distribution.append(self.desired_distribution.loc[member_id, DEFAULT_COL].item())

            for part in file_parts.values():
                choices: List[Worker] = choices.copy()
                workers: List[Worker] = np.random.choice(a=choices, p=desired_distribution, size=REPLICATION_LEVEL, replace=False)
                for worker in workers:
                    part.references += 1
                    worker.receive_part(part)

    def execute_epoch(self, epoch: int) -> None:
        """
        Orders all members to execute their epoch, i.e., perform stochastic swarm guidance for every file they hold
        If the Hive terminates early, the epoch's data is not added to FileData.SimulationData to avoid skewing previous results, when epoch causes failure early
        :param int epoch: simulation's current epoch
        :returns bool: false if Hive disconnected to persist the file it was responsible for, otherwise true is returned.
        """
        self.setup_epoch(epoch)
        try:
            offline_workers: List[Worker] = self.workers_execute_epoch()
            self.evaluate_hive_convergence()
            self.membership_maintenance(offline_workers)
            if epoch == MAX_EPOCHS:
                self.running = False
        except Exception as e:
            self.set_fail(f"Simulation failed due to unexpected exception, reason: {str(e)}")
        self.file.simulation_data.set_delay_at_index(self.set_recovery_epoch_sum, self.set_recovery_epoch_calls, self.current_epoch)

    def is_running(self) -> bool:
        return self.running

    def evaluate_hive_convergence(self):
        """
        Updates this epoch's distribution vector and compares it to the desired distribution vector to see if file distribution between members is near ideal
        and records epoch data accordingly.
        """
        if not self.members:
            self.set_fail("hive has no remaining members")

        parts_in_hive: int = 0
        for worker in self.members.values():
            if worker.status == Status.ONLINE:
                worker_parts_count = worker.get_file_parts_count(self.file.name)
                self.file.current_distribution.at[worker.id, DEFAULT_COL] = worker_parts_count
                parts_in_hive += worker_parts_count
            else:
                self.file.current_distribution.at[worker.id, DEFAULT_COL] = 0

        self.file.simulation_data.set_parts_at_index(parts_in_hive, self.current_epoch)

        if not parts_in_hive:
            self.set_fail("hive has no remaining parts")

        if self.file.equal_distributions(parts_in_hive):
            self.file.simulation_data.cswc_increment(1)
            self.file.simulation_data.try_append_to_convergence_set(self.current_epoch)
        else:
            self.file.simulation_data.save_sets_and_reset()

    # endregion

    # region Helpers

    def setup_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.corruption_chances[0] = np.log10(epoch).item() / 300.0
        self.corruption_chances[1] = 1.0 - self.corruption_chances[0]
        self.set_recovery_epoch_sum = 0
        self.set_recovery_epoch_calls = 0

    def workers_execute_epoch(self, lost_parts_count: int = 0) -> List[Worker]:
        """
        Orders all members of the hive to execute their epoch and updates some fields within SimulationData output file accordingly
        :param int lost_parts_count: zero
        :returns List[Worker] offline_workers: a populated list of offline workers.
        """
        offline_workers: List[Worker] = []
        for worker in self.members.values():
            if worker.get_epoch_status() == Status.ONLINE:
                worker.execute_epoch(self, self.file.name)  # do not forget, file corruption, can also cause Hive failure: see Worker.discard_part(...)
            else:
                lost_parts = worker.get_file_parts(self.file.name)
                lost_parts_count += len(lost_parts)
                offline_workers.append(worker)
                for part in lost_parts.values():
                    self.set_recovery_epoch(part)
                    if part.decrease_and_get_references() == 0:
                        self.set_fail("lost all replicas of file part with id: {}".format(part.id))
        if len(offline_workers) >= len(self.members):
            self.set_fail("all hive members disconnected simultaneously")
        self.file.simulation_data.set_disconnected_and_losses(len(offline_workers), lost_parts_count, self.current_epoch)
        return offline_workers

    def membership_maintenance(self, offline_workers: List[Worker]) -> None:
        """
        Used to ensure hive stability and proper swarm guidance behavior. No maintenance is needed if there are no disconnected workers in the inputed list.
        :param List[Worker] offline_workers: collection of members who disconnected during this epoch
        """
        # remove all disconnected workers from the hive
        for member in offline_workers:
            self.members.pop(member.id, None)

        damaged_hive_size = len(self.members)
        if damaged_hive_size >= self.sufficient_size:
            self.remove_cloud_reference()

        if damaged_hive_size >= self.redundant_size:
            status_before_recovery = "redundant"  # TODO: future-iterations evict worse members
        elif self.original_size <= damaged_hive_size < self.redundant_size:
            status_before_recovery = "stable"
        elif self.sufficient_size <= damaged_hive_size < self.original_size:
            status_before_recovery = "sufficient"
            self.members.update(self.__get_new_members())
        elif self.critical_size < damaged_hive_size < self.sufficient_size:
            status_before_recovery = "unstable"
            self.members.update(self.__get_new_members())
        elif 0 < damaged_hive_size <= self.critical_size:
            status_before_recovery = "critical"
            self.members.update(self.__get_new_members())
            self.add_cloud_reference()
        else:
            status_before_recovery = "dead"

        healed_hive_size = len(self.members)
        if damaged_hive_size != healed_hive_size:
            self.create_and_bcast_new_transition_matrix()

        self.file.simulation_data.set_membership_maintenace_at_index(status_before_recovery, damaged_hive_size, healed_hive_size, self.current_epoch)

    def __get_new_members(self) -> Dict[str, Worker]:
        return self.hivemind.find_replacement_worker(self.members, self.original_size - len(self.members))

    def set_fail(self, msg: str) -> None:
        self.running = False
        self.file.simulation_data.set_fail(self.current_epoch, msg)

    def tear_down(self, origin: str, epoch: int) -> None:
        # self.hivemind.append_epoch_results(self.id, self.file.simulation_data.__repr__()) TODO: future-iterations where Hivemind has multiple hives
        self.file.jwrite(self, origin, epoch)

    def set_recovery_epoch(self, part: SharedFilePart) -> None:
        self.set_recovery_epoch_sum += part.set_recovery_epoch(self.current_epoch)
        self.set_recovery_epoch_calls += 1

    def validate_transition_matrix(self, transition_matrix: pd.DataFrame, target_distribution: pd.DataFrame) -> bool:
        t_pow = np.linalg.matrix_power(transition_matrix.to_numpy(), 4096)
        column_count = t_pow.shape[1]
        for j in range(column_count):
            test_target = t_pow[:, j]  # gets array column j
            if not np.allclose(test_target, target_distribution[DEFAULT_COL].values, atol=1e-02):
                return False
        return True

    def create_and_bcast_new_transition_matrix(self):
        tries = 1
        result: pd.DataFrame = pd.DataFrame()
        while (tries <= 3):
            print(f"validating transition matrix... atempt: {tries}")
            result = self.new_transition_matrix()
            if self.validate_transition_matrix(result, self.desired_distribution):
                self.broadcast_transition_matrix(result)
                break
        self.broadcast_transition_matrix(result)  # if after 3 validations attempts no matrix was generated, use any other one.

    def select_fastest_topology(self, A: np.ndarray, v_: np.ndarray) -> np.ndarray:
        """
        Creates three possible transition matrices and selects the one that is theoretically faster to achieve the desired distribution v_
        :param np.ndarray A: An adjacency matrix that represents the network topology
        :param np.ndarray v_: A desired distribution vector that defines the returned matrix steady state property.
        :returns np.ndarray fastest_matrix: A markov transition matrix that converges to v_
        """
        results: List[Tuple[np.ndarray, float]] = [
            tmg.new_mh_transition_matrix(A, v_),
            tmg.new_sdp_mh_transition_matrix(A, v_),
            tmg.new_go_transition_matrix(A, v_)
        ]
        size = len(results)
        # fastest_matrix: np.ndarray = min(results, key=itemgetter(1))[0]
        min_mr = float('inf')
        fastest_matrix = None
        for i in range(size):
            i_mr = results[i][1]
            if i_mr < min_mr:
                min_mr = i_mr
                fastest_matrix = results[i][0]  # Worse case scenario fastest matrix will be the unoptmized MH transition matrix. Null checking thus, unneeded.

        size = fastest_matrix.shape[0]
        for j in range(size):
            fastest_matrix[:, j] = np.absolute(fastest_matrix[:, j])
            fastest_matrix[:, j] /= fastest_matrix[:, j].sum()
        return fastest_matrix

    # endregion

