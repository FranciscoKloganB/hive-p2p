from __future__ import annotations

import sys
import json
import math
import uuid
import traceback

import numpy as np
import pandas as pd
import utils.metropolis_hastings as mh

from enum import Enum
from random import randint
from globals.globals import *
from typing import Union, Dict, List, Any, Tuple
from domain.helpers.file_data import FileData

from utils import convertions, crypto, matrices
from utils.ResourceTracker import ResourceTracker as rT


class Status(Enum):
    """
    Enumerator class used to represent if a given Worker or super node is online, offline among other possible status
    """
    SUSPECT: int = 1
    OFFLINE: int = 2
    ONLINE: int = 3


class HttpCodes(Enum):
    """
    Enumerator class used to represent HTTP response codes
    """
    DUMMY: int = -1
    OK: int = 200
    BAD_REQUEST: int = 400
    NOT_FOUND: int = 404
    NOT_ACCEPTABLE: int = 406
    TIME_OUT: int = 408
    SERVER_DOWN: int = 521


class SharedFilePart:
    """
    Represents a simulation over the P2P Network that tries to persist a file using stochastic swarm guidance
    :ivar str id: concatenation of part_name | part_number
    :ivar str hive_id: uniquely identifies the hive that manages the shared file part instance
    :ivar str name: original name of the file this part belongs to
    :ivar int number: unique identifier for this file on the P2P network
    :ivar int references: indicates how many references exist for this SharedFilePart
    :ivar int recovery_epoch: indicates when recovery of this file will occur during
    :ivar str data: base64 string corresponding to the actual contents of this file part
    :ivar str sha256: hash value resultant of applying sha256 hash function over part_data param
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, hive_id: str, name: str, number: int, data: bytes):
        """
        Instantiates a SharedFilePart object
        :param str name: original name of the file this part belongs to
        :param int number: number that uniquely identifies this file part
        :param bytes data: Up to 2KB blocks of raw data that can be either strings or bytes
        """
        self.id: str = name + "_#_" + str(number)
        self.hive_id = hive_id
        self.name: str = name
        self.number: int = number
        self.references: int = REPLICATION_LEVEL
        self.recovery_epoch: int = sys.maxsize
        self.data: str = convertions.bytes_to_base64_string(data)
        self.sha256: str = crypto.sha256(self.data)
    # endregion

    # region Simulation Interface
    def set_epochs_to_recover(self, epoch: int) -> int:
        """
        Assigns a value to the instance's recovery_epoch attribute that indicates when a Worker who posses a reference to it, can replicate the part.
        :param int epoch: current simulation's epoch
        :returns int: expected delay
        """
        proposed_recovery_epoch: int = epoch + randint(MIN_DETECTION_DELAY, MAX_DETECTION_DELAY)
        if self.recovery_epoch < epoch or proposed_recovery_epoch < self.recovery_epoch:
            self.recovery_epoch = proposed_recovery_epoch

        if self.recovery_epoch == sys.maxsize:
            return 0

        return self.recovery_epoch - epoch

    def reset_epochs_to_recover(self) -> None:
        """
        Resets self.recovery_epoch attribute back to the default value of -1
        """
        self.recovery_epoch = sys.maxsize

    def can_replicate(self, epoch: int) -> int:
        """
        :param int epoch: current simulation's epoch
        :returns int: how many times the caller should replicate the SharedFilePart instance, if such action is possible
        """
        if self.references < REPLICATION_LEVEL and self.recovery_epoch - epoch <= 0:
            self.recovery_epoch = sys.maxsize  # Stops workers from repeatedly replicating this file part once one of them replicates
            return REPLICATION_LEVEL - self.references
        else:
            return 0
    # endregion

    # region Overrides
    def __str__(self):
        return "part_name: {},\npart_number: {},\npart_id: {},\npart_data: {},\nsha256: {}\n".format(self.name, self.number, self.id, self.data, self.sha256)
    # endregionss

    # region Helpers
    def decrease_and_get_references(self):
        self.references = self.references - 1
        return self.references

    def increase_and_get_references(self):
        self.references = self.references + 1
        return self.references
    # endregion


class Hivemind:
    """
    Representation of the P2P Network super node managing one or more hives ---- Simulator is piggybacked
    :ivar int max_epochs: number of stages the hive has to converge to the ddv before simulation is considered disconnected
    :ivar Dict[str, Hive] hives: collection mapping hives' uuid (attribute Hive.id) to the Hive instances
    :ivar Dict[str, Worker] workers: collection mapping workers' names to their Worker instances
    :ivar Dict[str, FileData] files_data: collection mapping file names on the system and their FileData instance
    :ivar Dict[Union[Worker, str], int] workers_status: maps workers or their names to their connectivity status
    :ivar Dict[str, List[FileData]] workers_hives: maps workers' names to hives they are known to belong to
    :ivar Dict[str, float] workers_uptime: maps workers' names to their expected uptime
    """

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, simfile_name: str) -> None:
        """
        Instantiates an Hivemind object
        :param str simfile_name: path to json file containing the parameters this simulation should execute with
        """
        self.epoch = 1
        self.results: Dict[str, Any] = {}

        simfile_path: str = os.path.join(SIMULATION_ROOT, simfile_name)
        with open(simfile_path) as input_file:
            json_obj: Any = json.load(input_file)

            # Init basic simulation variables
            self.hives: Dict[str, Hive] = {}
            self.workers: Dict[str, Worker] = {}

            # Instantiaite jobless Workers
            for worker_id, worker_uptime in json_obj['peers_uptime'].items():
                worker: Worker = Worker(worker_id, worker_uptime)
                self.workers[worker.id] = worker

            # Read and split all shareable files specified on the input, also assign Hive initial attributes (uuid, members, and FileData)
            hive: Hive
            files_spreads: Dict[str, str] = {}
            files_dict: Dict[str, Dict[int, SharedFilePart]] = {}
            file_parts: Dict[int, SharedFilePart]

            shared: Dict[str, Dict[str, Union[List[str], str]]] = json_obj['shared']
            for file_name in shared:
                with open(os.path.join(SHARED_ROOT, file_name), "rb") as file:
                    part_number: int = 0
                    file_parts = {}
                    files_spreads[file_name] = shared[file_name]['spread']
                    hive = self.__new_hive(shared, file_name)  # Among other things, assigns initial Hive members to the instance, implicitly set routing tables
                    while True:
                        read_buffer = file.read(READ_SIZE)
                        if read_buffer:
                            part_number = part_number + 1
                            file_parts[part_number] = SharedFilePart(hive.id, file_name, part_number, read_buffer)
                        else:
                            files_dict[file_name] = file_parts
                            break
                    hive.file.parts_count = part_number

            # Distribute files before starting simulation
            for hive in self.hives.values():
                hive.spread_files(files_spreads[hive.file.name], files_dict[hive.file.name])
    # endregion

    # region Simulation Interface
    def execute_simulation(self) -> None:
        """
        Runs a stochastic swarm guidance algorithm applied to a P2P network
        """
        failed_hives: List[str] = []
        while self.epoch < MAX_EPOCHS_PLUS and self.hives:
            print(self.epoch)
            for hive in self.hives.values():
                hive.execute_epoch(self.epoch)
                if not hive.is_running():
                    failed_hives.append(hive.id)
            for hive_id in failed_hives:
                self.hives.pop(hive_id)
                if not self.hives:
                    sys.exit("Simulation terminated at epoch {} because all hives disconnected before max epochs were reached".format(self.epoch))
            self.epoch += 1

    def append_epoch_results(self, hive_id: str, hive_results: [Dict, Any]) -> True:
        self.results[hive_id] = hive_results
    # endregion

    # region Keeper Interface
    def receive_complaint(self, suspects_name: str) -> None:
        """
        Registers a complaint on the named worker, if enough complaints are received, broadcasts proper action to all
        hives' workers to which the suspect belonged to.
        :param suspects_name: id of the worker which regards the complaint
        """
        # TODO future-iterations:
        #  1. register complaint
        #  2. when byzantine complaints > threshold
        #    2.1. find away of obtaining shared_file_names user had
        #    2.2. discover the files the node used to share, probably requires yet another sf_strucutre
        #    2.3. ask the next highest density node that is alive to rebuild dead nodes' files
        raise NotImplementedError()
    # endregion

    # region Peer Search and Cloud References
    def find_replacement_worker(self, exclusion_dict: Dict[str, Worker], quantity: int) -> Dict[str, Worker]:
        """
        Selects a worker who is at least as good as dead worker and updates FileData associated with the file
        :param Dict[str, Worker] exclusion_dict: collection of worker ids that the calling hive haves no interest in, for any reason
        :param int quantity: how many replacements the calling hive desires.
        :returns Dict[str, Worker] selected_workers: a collection of replacements a hive can use w/o guarantees that enough, if at all, replacements are found
        """
        selected_workers: Dict[str, Worker] = {}
        for worker in self.workers.values():
            if worker.status == Status.ONLINE and worker.id not in exclusion_dict:
                selected_workers[worker.id] = worker
                if len(selected_workers) == quantity:
                    return selected_workers
        return selected_workers

    def get_cloud_reference(self) -> str:
        """
        TODO: future-iteration
        :returns a cloud reference that can be used to persist files with more reliability
        """
        return ""
    # endregion

    # region Helpers
    def __new_hive(self, shared: Dict[str, Dict[str, Union[List[str], str]]], file_name: str) -> Hive:
        """
        Creates a new hive
        """
        hive_members: Dict[str, Worker] = {}
        for worker_id in shared[file_name]['members']:
            hive_members[worker_id] = self.workers[worker_id]
        hive = Hive(self, file_name, hive_members)
        self.hives[hive.id] = hive
        return hive
    # endregion


class Hive:
    """
    :ivar int current_epoch: tracks the epoch at which the Hive is currently at
    :ivar List[float, float] corruption_chances: used to simulate file corruption on behalf of the workers, to avoid keeping independant distributions for each part and each replica
    :ivar str id: unique identifier in str format
    :ivar Hivemind hivemind: reference to the master server, which in this case is just a simulator program
    :ivar FileData Union[None, FileData]: instance of class FileData which contains information regarding the file persisted by this hive
    :ivar Dict[str, Worker] members: Workers that belong to this P2P Hive, key is worker.id, value is the respective Worker instance
    :ivar int critical_size: minimum number of replicas required for data recovery plus the number of peer faults the system must support during replication.
    :ivar int sufficient_size: depends on churn-rate and equals critical_size plus the number of peers expected to fail between two successive recovery phases
    :ivar int redudant_size: application-specific system parameter, but basically represents that the hive is to big
    :ivar DataFrame desired_distribution: distribution hive members are seeking to achieve for each the files they persist together.
    """
    # region Class Variables, Instance Variables and Constructors
    def __init__(self, hivemind: Hivemind, file_name: str, members: Dict[str, Worker]) -> None:
        """
        Instantiates an Hive abstraction
        :param Hivemind hivemind: Hivemand instance object which leads the simulation
        :param str file_name: name of the file this Hive is responsible for
        :param Dict[str, Worker] members: collection mapping names of the Hive's initial workers' to their Worker instances
        """
        self.current_epoch: int = 0
        self.corruption_chances: List[float] = [0, 0]
        self.id: str = str(uuid.uuid4())
        self.hivemind = hivemind
        self.file: FileData = FileData(file_name)
        self.members: Dict[str, Worker] = members
        self.critical_size: int = REPLICATION_LEVEL
        self.sufficient_size: int = self.critical_size + math.ceil(len(self.members) * 0.34)
        self.original_size: int = len(members)
        self.redudant_size: int = self.sufficient_size + len(self.members)
        self.desired_distribution = None
        self.file.simulation_data.set_membership_maintenace_at_index(status="stable", size_before=len(members), size_after=len(members), i=0)
        self.running = True
        self.broadcast_transition_matrix(self.new_transition_matrix())  # implicitly inits self.desired_distribution within new_transition_matrix()
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

    def route_part(self, destination_name: str, part: SharedFilePart) -> Any:
        """
        Receives a shared file part and sends it to the given destination
        :param str destination_name: destination worker's id
        :param SharedFilePart part: the file part to send to specified worker
        :returns int: http codes based status of destination worker
        """
        self.file.simulation_data.set_moved_parts_at_index(1, self.current_epoch)
        if np.random.choice(a=TRUE_FALSE, p=COMMUNICATION_CHANCES):  # Simulates channel loss - Makes convergence harder
            self.file.simulation_data.set_lost_messages_at_index(1, self.current_epoch)
            return HttpCodes.TIME_OUT

        if np.random.choice(a=TRUE_FALSE, p=self.corruption_chances):  # File corruption, in a simplistic manner - Makes durability lower
            if part.decrease_and_get_references() == 0:
                return False
            self.file.simulation_data.set_corrupt_files_at_index(1, self.current_epoch)
            return HttpCodes.BAD_REQUEST

        member = self.members[destination_name]
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
        self.file.desired_distribution = self.desired_distribution
        self.file.current_distribution = pd.DataFrame(data=[0] * len(uptimes_normalized), index=member_ids)

        return uptimes_normalized

    def new_transition_matrix(self) -> pd.DataFrame:
        """
        returns DataFrame: Creates a new transition matrix for the members of the Hive, to be followed independently by each of them
        """
        desired_distribution: List[float]
        adjancency_matrix: List[List[int]]
        member_uptimes: List[float] = []
        member_ids: List[str] = []

        for worker in self.members.values():
            member_uptimes.append(worker.uptime)
            member_ids.append(worker.id)

        adjancency_matrix = matrices.new_symmetric_adjency_matrix(len(member_ids))
        desired_distribution = self.new_desired_distribution(member_ids, member_uptimes)

        transition_matrix: np.ndarray = mh.metropolis_algorithm(adjancency_matrix, desired_distribution, column_major_out=True)
        return pd.DataFrame(transition_matrix, index=member_ids, columns=member_ids)

    def broadcast_transition_matrix(self, transition_matrix: pd.DataFrame) -> None:
        """
        Gives each member his respective slice (vector column) of the transition matrix the Hive is currently executing.
        post-scriptum: we could make an optimization that sets a transition matrix for the hive, ignoring the file names, instead of mapping different file
        names to an equal transition matrix within each hive member, thus reducing space overhead arbitrarly, however, this would make Simulation harder. This
        note is kept for future reference. This also assumes an hive can store multiple files. For simplicity each Hive only manages one file for now.
        """
        transition_vector: pd.DataFrame
        for worker in self.members.values():
            transition_vector = transition_matrix.loc[:, worker.id]
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
                desired_distribution.append(self.desired_distribution.loc[member_id, DEFAULT_COLUMN].item())

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
        try:
            lost_parts_count, recoverable_parts, disconnected_workers = self.__setup_epoch(epoch)
            for worker in self.members.values():
                if worker.get_epoch_status() != Status.ONLINE:
                    lost_parts: Dict[int, SharedFilePart] = worker.get_file_parts(self.file.name)
                    lost_parts_count += len(lost_parts)
                    disconnected_workers.append(worker)
                    # Process data held by the disconnected worker
                    for number, part in lost_parts.items():
                        if part.decrease_and_get_references() <= 0:
                            self.__set_fail(epoch, "lost all replicas of at least one file part")
                            self.__tear_down()
                            return
                        recoverable_parts[number] = part
                else:
                    worker.execute_epoch(self, self.file.name)

            # Perfect failure detection, assumes that once a machine goes offline it does so permanently for all hives, so, pop members who disconnected
            if len(disconnected_workers) >= len(self.members):
                self.__set_fail(epoch, "all hive's workers disconnected at the same epoch")
                self.__tear_down()
                return

            self.file.simulation_data.set_disconnected_and_losses(disconnected=len(disconnected_workers), lost=lost_parts_count)
            self.evaluate_hive_convergence()

            status, size_before, size_after = self.__membership_maintenance(disconnected_workers)
            self.file.simulation_data.set_membership_maintenace_at_index(status, size_before, size_after, epoch)

            sum_delay = 0
            for part in recoverable_parts.values():
                sum_delay += part.set_epochs_to_recover(epoch)
            self.file.simulation_data.set_delay_at_index((sum_delay / len(recoverable_parts)) if recoverable_parts else 0, epoch)

            if epoch == MAX_EPOCHS:
                self.__tear_down()

        except Exception as e:
            self.__set_fail(epoch, "".join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
            self.__tear_down()

    def is_running(self) -> bool:
        return self.running

    def evaluate_hive_convergence(self):
        """
        Updates this epoch's distribution vector and compares it to the desired distribution vector to see if file distribution between members is near ideal
        and records epoch data accordingly.
        """
        parts_in_hive: int = 0
        for worker in self.members.values():
            if worker.status == Status.ONLINE:
                worker_parts_count = worker.get_file_parts_count(self.file.name)
                self.file.current_distribution.at[worker.id, DEFAULT_COLUMN] = worker_parts_count
                parts_in_hive += worker_parts_count
            else:
                self.file.current_distribution.at[worker.id, DEFAULT_COLUMN] = 0

        self.file.simulation_data.parts_in_hive[self.current_epoch] = parts_in_hive

        if not parts_in_hive:
            raise RuntimeError("No parts in hive")

        if self.file.equal_distributions(parts_in_hive):
            self.file.simulation_data.cswc_increment(1)
            self.file.simulation_data.try_append_to_convergence_set(self.current_epoch)
        else:
            self.file.simulation_data.save_sets_and_reset()
    # endregion

    # region Helpers
    def __membership_maintenance(self, disconnected_workers: List[Worker]) -> Tuple[str, int, int]:
        """
        Used to ensure hive stability and proper swarm guidance behavior. No maintenance is needed if there are no disconnected workers in the inputed list.
        :param List[Worker] disconnected_workers: collection of members who disconnected during this epoch
        :returns Tuple[str, int, int] (status_before_recovery, size_before_recovery, size_after_recovery)
        """
        # remove all disconnected workers from the hive
        for member in disconnected_workers:
            self.members.pop(member.id)

        damaged_hive_size = len(self.members)

        if damaged_hive_size >= self.sufficient_size:
            self.remove_cloud_reference()

        if damaged_hive_size >= self.redudant_size:
            status_before_recovery = "redundant"  # TODO: future-iterations evict worse members
        elif self.original_size <= damaged_hive_size < self.redudant_size:
            status_before_recovery = "stable"
        elif self.sufficient_size <= damaged_hive_size < self.original_size:
            status_before_recovery = "sufficient"
            self.members.update(self.__get_new_members())
        elif self.critical_size < damaged_hive_size < self.sufficient_size:
            status_before_recovery = "unstable"
            self.members.update(self.__get_new_members())
        else:
            status_before_recovery = "critical"
            self.add_cloud_reference()
            self.members.update(self.__get_new_members())

        healed_hive_size = len(self.members)
        if damaged_hive_size != healed_hive_size:
            self.broadcast_transition_matrix(self.new_transition_matrix())

        return status_before_recovery, damaged_hive_size, healed_hive_size

    def __get_new_members(self) -> Dict[str, Worker]:
        return self.hivemind.find_replacement_worker(self.members, self.original_size - len(self.members))

    def __setup_epoch(self, epoch: int) -> Tuple[int, Dict[int, SharedFilePart], List[Worker]]:
        self.current_epoch = epoch
        self.corruption_chances[0] = np.log10(epoch).item() / 100.0
        self.corruption_chances[1] = 1.0 - self.corruption_chances[0]
        return 0, {}, []

    def __set_fail(self, epoch: int, msg: str) -> bool:
        print(msg)
        return self.file.simulation_data.set_fail(epoch, msg)

    def __tear_down(self) -> None:
        # self.hivemind.append_epoch_results(self.id, self.file.simulation_data.__repr__()) TODO: future-iterations where Hivemind has multiple hives
        self.file.jwrite(self.file.simulation_data)
        self.running = False
    # endregion


class Worker:
    """
    Defines a worker node on the P2P network.
    :cvar List[Union[Status.ONLINE, Status.OFFLINE, Status.SUSPECT]] ON_OFF: possible Worker states, will be extended to include suspect down the line
    :ivar str id: unique identifier of the worker instance on the network
    :ivar float uptime: average worker uptime
    :ivar float disconnect_chance: (100.0 - worker_uptime) / 100.0
    :ivar Dict[str, Hive] hives: dictionary that maps the hive_ids' this worker instance belongs to, to the respective Hive instances
    :ivar Dict[str, Dict[int, SharedFilePart]] files: collection mapping file names to file parts and their contents
    :ivar Dict[str, pd.DataFrame] routing_table: collection mapping file names to the respective transition probabilities followed by the worker instance
    :ivar Union[int, Status] status: indicates if this worker instance is online or offline, might have other non-intuitive status, hence bool does not suffice
    """

    ON_OFF: List[Union[Status.ONLINE, Status.OFFLINE, Status.SUSPECT]] = [Status.ONLINE, Status.OFFLINE]

    # region Class Variables, Instance Variables and Constructors
    def __init__(self, worker_id: str, worker_uptime: float):
        self.id: str = worker_id
        self.uptime: float = worker_uptime
        self.disconnect_chance: float = 1.0 - worker_uptime
        self.hives: Dict[str, Hive] = {}
        self.files: Dict[str, Dict[int, SharedFilePart]] = {}
        self.routing_table: Dict[str, pd.DataFrame] = {}
        self.status: Union[int, Status] = Status.ONLINE
    # endregion

    # region Recovery
    def init_recovery_protocol(self, file_name: str) -> SharedFilePart:
        """
        Reconstructs a file and then splits it into globals.READ_SIZE before redistributing them to the rest of the hive
        :param str file_name: id of the shared file that needs to be reconstructed by the Worker instance
        """
        # TODO future-iterations:
        raise NotImplementedError()
    # endregion

    # region Routing Table
    def set_file_routing(self, file_name: str, transition_vector: Union[pd.Series, pd.DataFrame]) -> None:
        """
        Maps file id with state transition probabilities used for routing
        :param str file_name: a file id that is being shared on the hive
        :param Union[pd.Series, pd.DataFrame] transition_vector: probabilities of going from current worker to some worker on the hive
        """
        if isinstance(transition_vector, pd.Series):
            self.routing_table[file_name] = transition_vector.to_frame()
        elif isinstance(transition_vector, pd.DataFrame):
            self.routing_table[file_name] = transition_vector
        else:
            raise ValueError("Worker.set_file_routing expects a pandas.Series or pandas.DataFrame as type for transition vector parameter.")

    def remove_file_routing(self, file_name: str) -> None:
        """
        Removes a shared file's routing information from the routing table
        :param str file_name: id of the shared file whose routing information is being removed from routing_table
        """
        self.files.pop(file_name)
    # endregion

    # region File Routing
    def send_part(self, hive: Hive, part: SharedFilePart) -> Union[int, HttpCodes]:
        """
        Attempts to send a file part to another worker
        :param Hive hive: Gateway hive that will deliver this file to other worker
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        """
        routing_vector: pd.DataFrame = self.routing_table[part.name]
        hive_members: List[str] = [*routing_vector.index]
        member_chances: List[float] = [*routing_vector.iloc[:, DEFAULT_COLUMN]]
        destination: str = np.random.choice(a=hive_members, p=member_chances).item()  # converts numpy.str to python str
        if destination == self.id:
            return HttpCodes.DUMMY
        return hive.route_part(destination, part)

    def receive_part(self, part: SharedFilePart) -> int:
        """
        Keeps a new, single, shared file part, along the ones already stored by the Worker instance
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        :returns HttpCodes int
        """
        if part.name not in self.files:
            self.files[part.name] = {}  # init dict that accepts <key: id, value: sfp> pairs for the file

        if crypto.sha256(part.data) != part.sha256:
            return HttpCodes.BAD_REQUEST  # inform sender that his part is corrupt, don't initiate recovery protocol, to avoid denial of service attacks on worker
        elif part.number in self.files[part.name]:
            return HttpCodes.NOT_ACCEPTABLE  # reject repeated replicas even if they are correct
        else:
            self.files[part.name][part.number] = part
            return HttpCodes.OK  # accepted file part, because Sha256 was correct and Worker did not have this replica yet
    # endregion

    # region Swarm Guidance Interface
    def execute_epoch(self, hive: Hive, file_name: str) -> None:
        """
        For each part kept by the Worker instance, get the destination and send the part to it
        :param Hive hive: Hive instance that ordered execution of the epoch
        :param str file_name: the file parts that should be routed
        """
        file_cache: Dict[int, SharedFilePart] = self.files.get(file_name, {})
        epoch_cache: Dict[int, SharedFilePart] = {}
        for number, part in file_cache.items():
            self.try_replication(hive, part)
            response_code = self.send_part(hive, part)
            if response_code == HttpCodes.OK:
                pass  # self.files[file_name].pop(number), leave here for readability, but can't actually modify collection while iterating, pandora box!
            elif response_code == HttpCodes.BAD_REQUEST:
                # TODO: future-iterations, right now, assume replicas is lost -> part = self.init_recovery_protocol(part.name)
                pass  # self.files[file_name].pop(number), same as HttpCodes.OK, except destination worker did not keep this replica
            elif response_code != HttpCodes.OK:
                epoch_cache[number] = part  # This case includes HttpCodes.TIME_OUT and HttpCodes.NOT_ACCEPTABLE
        self.files[file_name] = epoch_cache  # In next epoch this worker instance only keeps files that were not accepted or were not corrupted
    # endregion

    def try_replication(self, hive: Hive, part: SharedFilePart) -> None:
        """
        Equal to send part but with different semantics, as file is not routed following swarm guidance, but instead by choosing the most reliable peers in the hive
        post-scriptum: This function is hacked... And should only be used for simulation purposes
        :param Hive hive: Gateway hive that will deliver this file to other worker
        :param SharedFilePart part: data class instance with data w.r.t. the shared file part and it's raw contents
        """
        replicate: int = part.can_replicate(hive.current_epoch)  # Number of times that file part needs to be replicated to achieve REPLICATION_LEVEL
        if replicate > 0:
            hive_member_ids: List[str] = [*hive.desired_distribution.sort_values(DEFAULT_COLUMN, ascending=False)]
            for member_id in hive_member_ids:
                if replicate == 0:
                    part.reset_epochs_to_recover()
                    break  # replication level achieved, no need to produce more copies
                elif member_id == self.id:
                    continue  # don't send to self, it would only get rejected
                elif hive.route_part(member_id, part) == HttpCodes.OK:
                    replicate -= 1  # decrease needed replicas
                    part.references += 1  # for each successful deliver increase number of copies in the hive
            if replicate > 0:
                part.recovery_epoch = hive.current_epoch + 1
    # region PSUtils Interface
    # noinspection PyIncorrectDocstring
    @staticmethod
    def get_resource_utilization(*args) -> Dict[str, Any]:
        """
        Obtains one ore more performance attributes for the Worker's instance machine
        :param *args: Variable length argument list. See below
        :keyword arg:
        :arg cpu: system wide float detailing cpu usage as a percentage,
        :arg cpu_count: number of non-logical cpu on the machine as an int
        :arg cpu_avg: average system load over the last 1, 5 and 15 minutes as a tuple
        :arg mem: statistics about memory usage as a named tuple including the following fields (total, available),
        expressed in bytes as floats
        :arg disk: get_disk_usage dictionary with total and used keys (gigabytes as float) and percent key as float
        :returns Dict[str, Any] detailing the usage of the respective key arg. If arg is invalid the value will be -1.
        """
        results: Dict[str, Any] = {}
        for arg in args:
            results[arg] = rT.get_value(arg)
        return results
    # endregion

    # region Overrides
    def __hash__(self):
        # allows a worker object to be used as a dictionary key
        return hash(str(self.id))

    def __eq__(self, other):
        return self.id == other

    def __ne__(self, other):
        return not(self == other)
    # endregion

    # region Helpers
    def get_file_parts(self, file_name: str) -> Dict[int, SharedFilePart]:
        """
        Gets collection of file parts that correspond to the named file
        :param str file_name: the file parts that caller wants to retrieve from this worker instance
        :returns Dict[int, SharedFilePart]: reference to a collection that maps part numbers to file parts
        """
        return self.files.get(file_name, {})

    def get_file_parts_count(self, file_name: str) -> int:
        """
        Counts how many parts the Worker instance has of the named file
        :param str file_name: the file parts that caller wants to count
        :returns int: number of parts from the named shared file currently on the Worker instance
        """
        return len(self.files.get(file_name, {}))

    def get_epoch_status(self) -> Union[Status.ONLINE, Status.OFFLINE, Status.SUSPECT]:
        """
        When called, the worker instance decides if it should switch status
        """
        if self.status != Status.ONLINE:
            return self.status
        else:
            epoch_status = np.random.choice(Worker.ON_OFF, p=[self.uptime, self.disconnect_chance])
            self.status = epoch_status
            return epoch_status
    # endregion
