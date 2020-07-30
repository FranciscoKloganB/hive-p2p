from __future__ import annotations

import os
import math
import uuid

import numpy as np
import pandas as pd
import domain.Hivemind as hm
import utils.matrices as matrices
import utils.transition_matrices as tmg

from domain.Worker import Worker
from typing import Dict, List, Any, Tuple, Optional
from domain.helpers.FileData import FileData
from domain.helpers.Enums import Status, HttpCodes
from domain.helpers.SharedFilePart import SharedFilePart
from domain.helpers.SimulationData import SimulationData
from globals.globals import REPLICATION_LEVEL, DEFAULT_COL, TRUE_FALSE, COMMUNICATION_CHANCES, MAX_EPOCHS

MATLAB_DIR = os.path.join(os.path.abspath(os.path.join(os.getcwd(), '..', 'app', 'scripts', 'matlabscripts')))


class Hive:
    """Represents a group of network nodes persisting a file.

    Notes:
        If you do not have a valid MatLab license you should comment
        all :py:attr:`~eng` related calls.

    Attributes:
        id:
            An uuid that uniquely identifies the Hive. Usefull for when
            there are multiple Hive instances in a simulation environment.
        current_epoch:
            The simulation's current epoch.
        corruption_chances:
            A two-element list containing the probability of file block replica
            being corrupted and not being corrupted, respectively. See
            :py:meth:`setup_epoch() <domain.Hive.Hive.setup_epoch>` for
            corruption chance configuration.
        desired_distribution (pandas DataFrame):
            Density distribution hive members must achieve with independent
            realizations for ideal persistence of the file.
        current_distribution (pandas DataFrame):
            Tracks the file current density distribution, updated at each epoch.
        hivemind:
            A reference to :py:class:`~domain.Hivemind.Hivemind` that
            coordinates this Hive instance.
        members:
            A collection of network nodes that belong to the Hive instance.
            See also :py:class:`~domain.domain.Worker`.
        file:
            A reference to :py:class:`~domain.helpers.FileData` object that
            represents the file being persisted by the Hive instance.
        critical_size:
            Minimum number of network nodes plus required to exist in the
            Hive to assure the target replication level.
        sufficient_size:
             Sum of :py:attr:`critical_size` and the number of nodes
             expected to fail between two successive recovery phases.
        original_size:
            The initial and optimal Hive size.
        redundant_size:
            Application-specific parameter, which indicates that membership
            of the Hive must be pruned.
        running:
            Indicates if the Hive instance is active. This attribute is
            used by :py:class:`~domain.Hivemind.Hivemind` to manage the
            simulation process.
        _recovery_epoch_sum:
            Helper attribute that facilitates the storage of the sum of the
            values returned by all :py:meth:`~SharedFilePart.set_recovery_epoch`
            method calls. Important for logging purposes.
        _recovery_epoch_calls:
            Helper attribute that facilitates the storage of the sum of the
            values returned by all :py:meth:`~SharedFilePart.set_recovery_epoch`
            method calls throughout the :py:attr:`~current_epoch`.
    """

    def __init__(self, hivemind: hm.Hivemind,
                 file_name: str,
                 members: Dict[str, Worker],
                 sim_id: int = 0,
                 origin: str = "") -> None:
        """Instantiates an Hive object

        Args:
            hivemind:
                A reference to an :py:class:`~domain.Hivemind.Hivemind`
                object that manages the Hive being initialized.
            file_name:
                The name of the file this Hive is responsible for persisting.
            members:
                A dictionary mapping unique identifiers to of the Hive's
                initial network nodes (:py:class:`~domain.domain.Worker`.)
                to their instance objects.
            sim_id:
                optional; Identifier that generates unique output file names,
                thus guaranteeing that different simulation instances do not
                overwrite previous out files.
            origin:
                optional; The name of the simulation file name that started
                the simulation process.
        """
        from matlab import engine as mleng
        print("Loading MatLab engine; This may take a few seconds...")
        self.eng = mleng.start_matlab()
        self.eng.cd(MATLAB_DIR)
        print("MatLab engine initiated. Resuming simulation...;")
        self.id: str = str(uuid.uuid4())
        self.current_epoch: int = 0
        self.current_distribution: pd.DataFrame = pd.DataFrame()
        self.desired_distribution: pd.DataFrame = pd.DataFrame()
        self.corruption_chances: List[float] = [0, 0]
        self.hivemind = hivemind
        self.members: Dict[str, Worker] = members
        self.file: FileData = FileData(file_name, sim_id=sim_id, origin=origin)
        self.critical_size: int = REPLICATION_LEVEL
        self.sufficient_size: int = self.critical_size + math.ceil(len(self.members) * 0.34)
        self.original_size: int = len(members)
        self.redundant_size: int = self.sufficient_size + len(self.members)
        self.running: bool = True
        self._recovery_epoch_sum: int = 0
        self._recovery_epoch_calls: int = 0
        self.create_and_bcast_new_transition_matrix()

    # endregion

    # region Routing

    def remove_cloud_reference(self) -> None:
        """Remove cloud references and delete files within it

        Notes:
            TODO: This method requires implementation at the user descretion.
        """
        pass

    def add_cloud_reference(self) -> None:
        """Adds a cloud server reference to the membership.

        This method is used when Hive membership size becomes compromised
        and a backup solution using cloud approaches is desired. The idea
        is that surviving members upload their replicas to the cloud server,
        e.g., an Amazon S3 instance. See Hivemind method
        :py:meth:`~domain.Hivemind.Hivemind.get_cloud_reference` for more
        details.

        Notes:
            TODO: This method requires implementation at the user descretion.
        """
        pass
        # noinspection PyUnusedLocal
        cloud_ref: str = self.hivemind.get_cloud_reference()

    def route_part(self,
                   sender: str,
                   destination: str,
                   part: SharedFilePart,
                   fresh_replica: bool = False) -> Any:
        """Sends one file block replica to some other network node.

        Args:
            sender:
                An identifier of the network node who is sending the message.
            destination:
                The destination network node identifier.
            part:
                The file block replica send to specified destination.
            fresh_replica:
                optional; Prevents recently created replicas from being
                corrupted, since they are not likely to be corrupted in disk.
                This argument facilitates simulation. (default: False)

        Returns:
            An HTTP code sent by destination network node.
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

    def new_desired_distribution(self,
                                 member_ids: List[str],
                                 member_uptimes: List[float]) -> List[float]:
        """Sets a new desired distribution for the Hive instance.

        Normalizes the received uptimes to create a stochastic representation
        of the desired distribution, which can be used by the different
        transition matrix generation strategies.

        Args:
            member_ids:
                A list of network node identifiers currently belonging
                to the Hive membership.
            member_uptimes:
                A list in which each index contains the uptime of the network
                node with the same index in `member_ids`.

        Returns:
            A list of floats with normalized uptimes which represent the
            'reliability' of network nodes.
        """
        uptime_sum = sum(member_uptimes)
        uptimes_normalized = \
            [member_uptime / uptime_sum for member_uptime in member_uptimes]

        v_ = pd.DataFrame(data=uptimes_normalized, index=member_ids)
        self.desired_distribution = v_
        cv_ = pd.DataFrame(data=[0] * len(v_), index=member_ids)
        self.current_distribution = cv_

        return uptimes_normalized

    def new_transition_matrix(self) -> pd.DataFrame:
        """Creates a new transition matrix to be distributed among hive members.

        Returns:
            The labeled matrix that has the fastests mixing rate from all
            the pondered strategies.
        """
        member_uptimes: List[float] = []
        member_ids: List[str] = []

        for worker in self.members.values():
            member_uptimes.append(worker.uptime)
            member_ids.append(worker.id)

        A: np.ndarray = np.asarray(
            matrices.new_symmetric_adjency_matrix(len(member_ids)))
        v_: np.ndarray = np.asarray(
            self.new_desired_distribution(member_ids, member_uptimes))

        T = self.select_fastest_topology(A, v_)

        return pd.DataFrame(T, index=member_ids, columns=member_ids)

    def broadcast_transition_matrix(self, m: pd.DataFrame) -> None:
        """Slices a transition matrix and delivers them to respective network nodes.

        Gives each member his respective slice (vector column) of the
        transition matrix the Hive is currently executing.

        Args:
            m:
                A transition matrix to be broadcasted to the network nodes
                belonging who are currently members of the Hive instance.

        Note:
            An optimization could be made that configures a transition matrix
            for the hive, independent of of file names, i.e., turn Hive
            groups into groups persisting multiple files instead of only one,
            thus reducing simulation spaceoverheads and in real-life
            scenarios, decreasing the load done to metadata servers, through
            queries and matrix calculations. For simplicity of implementation
            each Hive only manages one file for now.
        """
        for worker in self.members.values():
            transition_vector: pd.DataFrame = m.loc[:, worker.id]
            worker.set_file_routing(self.file.name, transition_vector)

    # endregion

    # region Simulation Interface

    # noinspection DuplicatedCode
    def spread_files(
            self, spread_mode: str, file_parts: Dict[int, SharedFilePart]
    ) -> None:
        """Spreads files over the initial members of the Hive
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
            self.set_fail(f"Exception caused simulation termination: {str(e)}")

        self.file.simulation_data.set_delay_at_index(self._recovery_epoch_sum,
                                                     self._recovery_epoch_calls,
                                                     self.current_epoch)

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
                self.current_distribution.at[worker.id, DEFAULT_COL] = worker_parts_count
                parts_in_hive += worker_parts_count
            else:
                self.current_distribution.at[worker.id, DEFAULT_COL] = 0

        self.file.simulation_data.set_parts_at_index(parts_in_hive, self.current_epoch)

        if not parts_in_hive:
            self.set_fail("hive has no remaining parts")

        self.file.parts_in_hive = parts_in_hive
        if self.equal_distributions():
            self.file.simulation_data.register_convergence(self.current_epoch)
        else:
            self.file.simulation_data.save_sets_and_reset()

    # endregion

    # region Helpers

    def equal_distributions(self) -> bool:
        """Infers if desired_distribution and current_distribution are equal.

        Equalility is calculated given a tolerance value calculated by
        FileData method defined at :py:method:`~new_tolerance() <FileData.new_tolerance>`.

        Returns:
            True if distributions are close enough to be considered equal,
            otherwise, it returns False.
        """
        if self.file.parts_in_hive == 0:
            return False

        size = len(self.current_distribution)
        tolerance = self.new_tolerance()
        for i in range(size):
            a = self.current_distribution.iloc[i, DEFAULT_COL]
            b = self.desired_distribution.iloc[i, DEFAULT_COL] * self.file.parts_in_hive
            if np.abs(a - np.ceil(b)) > tolerance:
                return False
        return True

    def new_tolerance(self) -> np.float64:
        """Calculates a tolerance value for the current epoch of the simulation.

        The tolerance is given by the maximum value in the desired
        distribution minus the minimum value times the numbers of parts,
        including replicas.

        Returns:
            The tolerance for the current epoch.
        """
        max_value = self.desired_distribution[DEFAULT_COL].max()
        min_value = self.desired_distribution[DEFAULT_COL].min()
        return np.ceil(np.abs(max_value - min_value)) * self.file.parts_in_hive

    def setup_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
        self.corruption_chances[0] = np.log10(epoch).item() / 300.0
        self.corruption_chances[1] = 1.0 - self.corruption_chances[0]
        self._recovery_epoch_sum = 0
        self._recovery_epoch_calls = 0

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
                    if part.decrement_and_get_references() == 0:
                        self.set_fail("lost all replicas of file part with id: {}".format(part.id))
        if len(offline_workers) >= len(self.members):
            self.set_fail("all hive members disconnected simultaneously")

        e: int = self.current_epoch
        sf: SimulationData = self.file.simulation_data
        sf.set_disconnected_workers_at_index(len(offline_workers), e)
        sf.set_lost_parts_at_index(lost_parts_count, e)

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
        self._recovery_epoch_sum += part.set_recovery_epoch(self.current_epoch)
        self._recovery_epoch_calls += 1

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
        while tries <= 3:
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
            tmg.new_go_transition_matrix(A, v_),
            tmg.go_with_matlab_bmibnb_solver(A, v_, self.eng)
        ]
        size = len(results)
        # fastest_matrix: np.ndarray = min(results, key=itemgetter(1))[0]
        min_mr = float('inf')
        fastest_matrix = None
        for i in range(size):
            i_mr = results[i][1]
            if i_mr < min_mr:
                print(f"currently selected matrix {i}")
                min_mr = i_mr
                fastest_matrix = results[i][0]  # Worse case scenario fastest matrix will be the unoptmized MH transition matrix. Null checking thus, unneeded.

        size = fastest_matrix.shape[0]
        for j in range(size):
            fastest_matrix[:, j] = np.absolute(fastest_matrix[:, j])
            fastest_matrix[:, j] /= fastest_matrix[:, j].sum()
        return fastest_matrix

    # endregion

