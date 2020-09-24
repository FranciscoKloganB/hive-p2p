from __future__ import annotations

from typing import Dict, Union

import domain.master_servers as ms
import domain.cluster_groups as cg
import domain.network_nodes as nn
import domain.helpers.enums as e
import domain.helpers.smart_dataclasses as sd

NodeDict: Dict[str, NodeType]
ClusterDict: Dict[str, ClusterType]
ReplicasDict: Dict[int, sd.FileBlockData]
HttpResponse: Union[int, e.HttpCodes]

MasterType: Union[
    ms.Master,
    ms.HiveMaster,
    ms.HDFSMaster,
    ms.NewscastMaster
]

ClusterType: Union[
    cg.Cluster,
    cg.HiveCluster,
    cg.HiveClusterExt,
    cg.HDFSCluster,
    cg.NewscastCluster
]

NodeType: Union[
    nn.Node,
    nn.HiveNode,
    nn.HiveNodeExt,
    nn.HDFSNode,
    nn.NewscastNode
]
