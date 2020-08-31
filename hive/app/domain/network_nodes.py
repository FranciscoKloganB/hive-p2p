"""This module contains domain specific classes that represent network nodes
responsible for the storage of :py:class:`file blocks
<app.domain.helpers.smart_dataclasses.FileBlockData>`. These could be
reliable servers or P2P nodes."""
from __future__ import annotations

import math
import sys
import traceback
from typing import Union, Dict, List

import domain.helpers.smart_dataclasses as sd
import domain.helpers.enums as e
import domain.master_servers as ms
import type_hints as th
import pandas as pd
import numpy as np

from utils import crypto
from environment_settings import TRUE_FALSE


class Node:
    """This class contains basic network node functionality that should
    always be useful.

    Attributes:
        id:
            A unique identifier for the ``Node`` instance.
        uptime:
            The amount of time the ``Node`` is expected to remain online
            without disconnecting. Current uptime implementation is based on
            availability percentages. Furthermore, when a HiveNode joins
            an Cluster as a replacement for some other HiveNode, in
            :py:meth:`app.domain.cluster_groups.Cluster._membership_maintenance`
            , the time the worker has been online is not considered. Thus,
            if a HiveNode belongs to only one Cluster he is guaranteed to
            remain online for exactly `uptime` *
            :py:attr:`environment_settings.MAX_EPOCHS`. If he belongs to
            multiple Hives in the simulation, than there is a possibility
            that he may go offline earlier.
        status:
            Indicates if the ``Node`` instance is online or offline. In later
            releases this could also contain a 'suspect' status. See
            :py:class:`app.domain.helpers.enums.Status`.
        suspicious_replies:
            Set that contains :py:class:`app.domain.helpers.enums.HttpCodes`
            that trigger complaints to monitors.
        files:
            A dictionary mapping file names to file block identifiers and their
            respective contents. This collection represents the file block
            blocks that are currently hosted in the HiveNode instance.
    """
    def __init__(self, uid: str, uptime: float) -> None:
        """Instantiates a HiveNode object.

        Workers are the network nodes responsible for persisting file block
        blocks.

        Args:
            uid:
                An unique identifier for the worker instance.
            uptime:
                The availability of the worker instance.
        """
        if uptime == 1.0:
            uptime = float('inf')
        else:
            uptime = math.floor(uptime * ms.Master.MAX_EPOCHS)

        self.id: str = uid
        self.uptime: float = uptime
        self.status: int = e.Status.ONLINE
        self.suspicious_replies = {
            e.HttpCodes.NOT_FOUND,
            e.HttpCodes.TIME_OUT,
            e.HttpCodes.SERVER_DOWN,
        }
        self.files: Dict[str, th.ReplicasDict] = {}

    # region Simulation steps
    def execute_epoch(self, cluster: th.ClusterType, fid: str) -> None:
        """Instructs the HiveNode instance to execute the epoch."""
        raise NotImplementedError("All children of class Node must "
                                  "implement their own execute_epoch.")
    # endregion

    # region File block management
    def receive_part(self, replica: sd.FileBlockData) -> int:
        """Endpoint for file block replica reception.

        Invoking this method results in the worker keeping store a new
        file block replica in his :py:attr:`files` collection, along the ones
        already there.

        Args:
            replica:
               The file block container to be received by the worker instance.

        Returns:
             An HTTP code defined in
             :py:class:`domain.helpers.enums.HttpCodes`. If upon integrity
             verification the sha256 hashvalue differs from the expected,
             the worker replies with a BAD_REQUEST code. If the HiveNode already
             owns a replica with the same number identifier it
             replies with NOT_ACCEPTABLE. Otherwise it replies with a OK,
             i.e., the delivery is successful.
        """
        if replica.name not in self.files:
            # init dict that accepts <key: id, value: sfp> pairs for the file
            self.files[replica.name] = {}

        if crypto.sha256(replica.data) != replica.sha256:
            # inform sender that his part is corrupt,
            # don't initiate recovery protocol - avoid DoS at current worker.
            return e.HttpCodes.BAD_REQUEST
        elif replica.number in self.files[replica.name]:
            # reject repeated blocks even if they are correct
            return e.HttpCodes.NOT_ACCEPTABLE
        else:
            # accepted file part
            self.files[replica.name][replica.number] = replica
            return e.HttpCodes.OK

    def replicate_part(
            self, cluster: th.ClusterType, replica: sd.FileBlockData) -> None:
        """Tries to restore the `replica`'s
        :py:const:`environment_settings.REPLICATION_LEVEL`.

        Similar to :py:meth:`domain.network_nodes.Node.send_part` but with
        slightly different instructions. In particular newly replicate
        replicas can not be corrupted at the current node, at the current epoch.

        Args:
            cluster:
                Gateway Cluster that will deliver the file block replica to
                some destination HiveNode.
            replica:
                The file block replica to be delivered.
        """
        raise NotImplementedError("")

    def send_part(self,
                  cluster: th.ClusterType,
                  destination: str,
                  replica: sd.FileBlockData) -> th.HttpResponse:
        """Attempts to send a replica to some other network node.

        Args:
            cluster:
                :py:mod:`Cluster Group Monitor <domain.cluster_groups>` that
                will act as a courier for the message to be sent. In a real
                world implementation this argument would not be needed and
                would not even make sense, but we use it to facilitate
                simulation management and environment logging.
            destination:
                The name, address or another unique identifier of the node
                that will receive the file block `replica`.
            replica:
                The file block container to be sent to some other worker.

        Returns:
             An HTTP code defined in
             :py:class:`domain.helpers.enums.HttpCodes` and the destination
             the part was sent to.
        """
        return cluster.route_part(self.id, destination, replica)

    def discard_part(self,
                     fid: str,
                     number: int,
                     corrupt: bool = False,
                     cluster: th.ClusterType = None) -> None:
        """Safely deletes a part from the HiveNode instance's disk.

        Args:
            fid:
                Name of the file the file block replica belongs to.
            number:
                The part number that uniquely identifies the file block.
            corrupt:
                If discard is being invoked due to identified file
                block corruption, e.g., Sha256 does not match the expected.
            cluster:
                Gateway Cluster that will set the recovery epoch (see
                :py:meth:`domain.cluster_groups.Cluster.set_recovery_epoch`
                or mark the simulation as failed.
        """
        replica: sd.FileBlockData = self.files.get(fid, {}).pop(number, None)
        if replica and corrupt:
            if replica.decrement_and_get_references() > 0:
                cluster.set_replication_epoch(replica)
            else:
                cluster._set_fail(f"Lost last file block replica with id "
                                  f"{replica.id} due to corruption.")
    # endregion

    # region Get methods
    def get_file_parts(self, fid: str) -> th.ReplicasDict:
        """Gets collection of file parts that correspond to the named file.

        Args:
            fid:
                The identifier that designates the file block blocks
                the caller wishes to obtain from the HiveNode instance.

        Returns:
             A collection that maps file block identifiers to file block
             containers.
        """
        return self.files.get(fid, {})

    def get_file_parts_count(self, fid: str) -> int:
        """Counts the number of file block blocks owned by the HiveNode
        for a given file identifier.

        Args:
             fid:
                An identifier of the file caller wishes to count.
        Returns:
            The number of file block blocks from the named file the HiveNode
            instance possesses.
        """
        return len(self.files.get(fid, {}))

    def get_status(self) -> int:
        """Used to obtain the status of the worker.

        This method simulates a ping. When invoked, the HiveNode instance
        decides if it should switch its status from ONLINE to some other
        depending on the time it has been active in the Cluster.

        Returns:
            The status of the worker. See
            :py:class:`domain.helpers.enums.e.Status`.
        """
        if self.status == e.Status.ONLINE:
            self.uptime -= 1
            if self.uptime <= 0:
                self.status = e.Status.OFFLINE
        return self.status
    # endregion

    # region Python dunder methods' overrides
    def __hash__(self):
        """Can use Node instance or id as dictionary key."""
        return hash(str(self.id))

    def __eq__(self, other):
        """Node equality is based solely on HiveNode id."""
        return self.id == other

    def __ne__(self, other):
        return not(self == other)
    # endregion


class HiveNode(Node):
    """Represents a network node that executes a Swarm Guidance algorithm.

    Workers work in one or more cluster_groups
    (:py:class:`domain.cluster_groups.Cluster`) to try to ensure the
    durability of files belonging to the distributed backup system. Their
    route file block blocks to other network_nodes at every epoch
    time according to the routing table of the file that block belongs to,
    i.e., according to the column vector of the transition matrix that is
    generated by
    (:py:meth:`domain.cluster_groups.Cluster.new_transition_matrix`).
    If every worker in the Cluster shares their blocks with other members by
    following their respective vector then eventually, the whole
    Cluster will reach the desired steady state distribution of file
    block blocks (:py:attr:`domain.cluster_groups.Cluster.v_`).

    Attributes:
        hives:
            A collection of :py:class:`domain.cluster_groups.Cluster` this
            HiveNode is a member of.
        routing_table:
            Contains the information required to appropriately route file
            block blocks to other HiveNode instances.
    """
    def __init__(self, uid: str, uptime: float) -> None:
        """Instantiates a HiveNode object.

        Workers are the network nodes responsible for persisting file block
        blocks.

        Args:
            uid:
                An unique identifier for the worker instance.
            uptime:
                The availability of the worker instance.
        """
        super().__init__(uid, uptime)
        self.hives: Dict[str, th.ClusterType] = {}
        self.routing_table: Dict[str, pd.DataFrame] = {}

    # region Simulation steps
    def execute_epoch(self, cluster: th.ClusterType, fid: str) -> None:
        """Instructs the HiveNode instance to execute the epoch.

        The method iterates all file block blocks in :py:attr:`files` and
        independently decides if they should be sent to other HiveNode
        instances by following :py:attr:`routing_table` column vectors.

        When a file block is sent to some other HiveNode a reply is
        awaited. In real world environments this should be assynchronous,
        but for simulation purposes it's synchronous and instantaneous. When the
        destination replies with OK, meaning it accepted the replica,
        this HiveNode instance deletes the replica from his disk. If it
        replies with a BAD_REQUEST the replica is discarded and the worker
        starts a recovery process in the Cluster. Any other code response
        results in the HiveNode instance keeping replica in his disk
        for at least one more epoch times. See
        :py:class:`domain.helpers.enums.HttpCodes` for more information on
        possible HTTP Codes.

        Overrides:
            :py:meth:`app.domain.network_nodes.Node.execute_epoch`.

        Args:
            cluster:
                Cluster instance that ordered execution of the epoch.
            fid:
                The identifier that determines which file blocks
                blocks should be routed.
        """
        file_view: th.ReplicasDict = self.files.get(fid, {}).copy()
        for number, replica in file_view.items():
            self.replicate_part(cluster, replica)
            destination = self.select_destination(replica.name)
            response_code = self.send_part(cluster, destination, replica)
            if response_code == e.HttpCodes.OK:
                self.discard_part(fid, number)
            elif response_code == e.HttpCodes.BAD_REQUEST:
                self.discard_part(fid, number, corrupt=True, cluster=cluster)
            elif response_code in self.suspicious_replies:
                cluster.complain(self.id, destination, response_code)
            # Else keep file part for at least one more epoch
    # endregion

    # region File block management
    def replicate_part(
            self, cluster: th.ClusterType, replica: sd.FileBlockData) -> None:
        """Tries to restore the `replica`'s
        :py:const:`~environment_settings.REPLICATION_LEVEL`.

        The file block replica is sent selectively in descending order to the
        most reliable Nodes in the Cluster down to the least
        reliable. Whereas
        :py:meth:`app.domain.network_nodes.HiveNode.send_part`. follows
        stochastic swarm guidance routing.

        Overrides:
            :py:meth:`app.domain.network_nodes.Node.replicate_part`.
        """
        # Number of times the block needs to be replicated.
        lost_replicas: int = replica.can_replicate(cluster.current_epoch)
        if lost_replicas > 0:
            sorted_members = [*cluster.v_.sort_values(0, ascending=False).index]
            for destination in sorted_members:
                if lost_replicas == 0:
                    break
                code = cluster.route_part(
                    self.id, destination, replica, fresh_replica=True)
                if code == e.HttpCodes.OK:
                    lost_replicas -= 1
                    replica.references += 1
                elif code in self.suspicious_replies:
                    cluster.complain(self.id, destination, code)
            # replication level may have not been completely restored
            replica.update_epochs_to_recover(cluster.current_epoch)
    # endregion

    # region Routing table management
    def set_file_routing(
            self, fid: str, transition_vector: Union[pd.Series, pd.DataFrame]
    ) -> None:
        """Maps a file id with a transition column vector used for routing.

        Args:
            fid:
                The identifier of the file the Cluster is persisting.
            transition_vector:
                A column vector with probabilities that dictate the odds of
                sending file block blocks belonging to the file with
                specified id to other Cluster members also working on the
                persistence of the file block blocks.

        Raises:
            ValueError:
                If ``transition_vector`` is not a pandas DataFrame and cannot
                be directly converted to it.

    """
        if isinstance(transition_vector, pd.Series):
            self.routing_table[fid] = transition_vector.to_frame()
        elif isinstance(transition_vector, pd.DataFrame):
            self.routing_table[fid] = transition_vector
        else:
            raise ValueError("set_file_routing method expects a pandas.Series ",
                             "or pandas.DataFrame as transition vector type.")

    def remove_file_routing(self, fid: str) -> None:
        """Removes a file id from the routing table.

        This method is called when a HiveNode is evicted from the Cluster
        and results in the complete deletion of all file blocks with that
        file id.

        Args:
            fid:
                Id of the file whose routing information is being
                removed from routing_table.
        """
        self.routing_table.pop(fid, pd.DataFrame())
        self.files.pop(fid, {})
    # endregion

    # region Swarm guidance
    def select_destination(self, fid: str) -> str:
        """Selects a random message destination according to `routing_table`
        probabilities for the specified file name.

        Args:
            fid:
                The unique file identifier.

        Returns:
            The name or address of the selected destination.
        """
        routing_vector: pd.DataFrame = self.routing_table[fid]
        hive_members: List[str] = [*routing_vector.index]
        member_chances: List[float] = [*routing_vector.iloc[:, 0]]
        try:
            return np.random.choice(a=hive_members, p=member_chances).item()
        except ValueError as vE:
            print(f"{routing_vector}\nStochastic?: {np.sum(member_chances)}")
            sys.exit("".join(
                traceback.format_exception(
                    etype=type(vE), value=vE, tb=vE.__traceback__)))
    # endregion


class HiveNodeExt(HiveNode):
    """Represents a network node that executes a Swarm Guidance algorithm.

    HiveNodeExt instances differ from HiveNode in the sense that the
    :py:class:`HiveClusterExt Nodes <domain.domain.HiveNode>` do not monitor their
    groups' peers, concerning suspicious behaviors.
    """

    # region Get methods
    def get_status(self) -> int:
        """Used to obtain the status of the worker.

        Overrides:
            :py:meth:`app.domain.network_nodes.Node.get_status`.
            When not ONLINE the node sets itself as Suspect and will only become
            offline when the monitor marks him as so (e.g.: due to complaints).

        Returns:
            The status of the worker. See
            :py:class:`app.domain.helpers.enums.Status`.
        """
        if self.status == e.Status.ONLINE:
            self.uptime -= 1
            if self.uptime <= 0:
                print(f"    [x] {self.id} now offline (suspect status).")
                self.status = e.Status.SUSPECT
        return self.status
    # endregion


class HDFSNode(Node):
    """Represents a data node in the Hadoop Distribute File System."""
    def __init__(self, uid: str, uptime: float) -> None:
        """Instantiates a HDFSNode object.

        These represent hadoop distributed file system data nodes.

        Args:
            uid:
                An unique identifier for the worker instance.
            uptime:
                The availability of the worker instance.
        """
        super().__init__(uid, uptime)

    # region Simulation steps
    def execute_epoch(self, cluster: th.ClusterType, fid: str) -> None:
        """Instructs the Node instance to execute the epoch.

        The method iterates all file block blocks in :py:attr:`~files` and
        silentry tries to corrupt them. In HDFS files' Sha256 is only
        verified when a user or client accesses the remote replica. Thus,
        no recovery epoch is not setted up when a corruption occurs. The
        corruption is still logged in the output file.

        Overrides:
            :py:meth:`app.domain.network_nodes.Node.execute_epoch`.

        Args:
            cluster:
                Cluster instance that ordered execution of the epoch.
            fid:
                The identifier that determines which file blocks
                blocks should be routed.
        """
        file_view: th.ReplicasDict = self.files.get(fid, {}).copy()
        for number, replica in file_view.items():
            self.replicate_part(cluster, replica)
            if np.random.choice(a=TRUE_FALSE, p=cluster.corruption_chances):
                # Don't set corrupt flag to ``True``, doing so causes
                # set_recovery_epoch to be called. HDFS Corruption is silent.
                self.discard_part(fid, number)
                # Log the corruption in output file.
                epoch = cluster.current_epoch
                cluster.file.logger.log_corrupted_file_blocks(1, epoch)
    # endregion

    # region File block management
    def replicate_part(
            self, cluster: th.ClusterType, replica: sd.FileBlockData) -> None:
        """Tries to restore the `replica`'s
        :py:const:`~environment_settings.REPLICATION_LEVEL`.

        The file block replica is sent selectively in descending order to the
        most reliable Nodes in the Cluster down to the least reliable.

        Overrides:
            :py:meth:`app.domain.network_nodes.Node.replicate_part`.
        """
        # Number of times the block needs to be replicated.
        lost_replicas: int = replica.can_replicate(cluster.current_epoch)
        if lost_replicas > 0:
            choices = [*cluster.members.values()]
            choices.sort(key=lambda node: node.uptime, reverse=True)
            for destination in choices:
                if lost_replicas == 0:
                    break
                code = cluster.route_part(
                    self.id, destination, replica, fresh_replica=True)
                if code == e.HttpCodes.OK:
                    lost_replicas -= 1
                    replica.references += 1
            # replication level may have not been completely restored
            replica.update_epochs_to_recover(cluster.current_epoch)
    # endregion

    # region Get Methods
    def get_status(self) -> int:
        """Used to obtain the status of the worker.

        Overrides:
            :py:meth:`app.domain.network_nodes.Node.get_status`.
            When not ONLINE the node sets itself as Suspect and will only become
            offline when the monitor marks him as so (e.g.: due to complaints).

        Returns:
            The status of the worker. See
            :py:class:`app.domain.helpers.enums.Status`.
        """
        if self.status == e.Status.ONLINE:
            self.uptime -= 1
            if self.uptime <= 0:
                print(f"    [x] {self.id} now offline (suspect status).")
                self.status = e.Status.SUSPECT
        return self.status
    # endregion
