from typing import List, NamedTuple, Optional, Sequence

import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize

import scml

__all__ = ["GreedyModularityCommunities", "AsyncLpaCommunities"]

log = scml.get_logger(__name__)


class GraphClustering:

    class ClusterInfo(NamedTuple):
        cluster_id: int
        # Count of distinct objects in the cluster, excluding duplicates.
        distinct_size: int
        # Count of all objects in the cluster, including duplicates.
        total_size: int

    class Result(NamedTuple):
        cluster_assignment: List[int]
        clusters: List["GraphClustering.ClusterInfo"]

    def __init__(
        self,
        edge_attr: str = "weight",
    ):
        self.edge_attr = edge_attr

    def _graph(
        self,
        embeddings: np.ndarray,
        score_cutoff: float,
        dtype=np.float16,
    ) -> nx.Graph:
        # L2 normalize to get unit vectors
        X = normalize(embeddings, norm="l2", axis=1)
        n = X.shape[0]
        adj = np.full((n, n), 1, dtype=dtype)
        for i in range(n):
            for j in range(i + 1, n):
                # vectors already L2 norm, so just take dot product
                cos_sim: float = np.dot(X[i], X[j]).item()
                # prune edges below threshold
                if cos_sim < score_cutoff:
                    cos_sim = 0
                adj[i][j] = cos_sim
                adj[j][i] = cos_sim
        # undirected weighted graph
        res: nx.Graph = nx.from_numpy_array(A=adj, parallel_edges=False, edge_attr=self.edge_attr)  # type: ignore[call-overload]
        return res

    def __call__(
        self, embeddings: np.ndarray, score_cutoff: float, *args, **kwargs
    ) -> Result:
        raise NotImplementedError("Implement this method in subclass")

    def _postprocess(
        self,
        clusters: Sequence,
        embeddings: np.ndarray,
        item_counts: Optional[List[int]] = None,
    ) -> Result:
        if item_counts is not None and len(item_counts) != int(embeddings.shape[0]):
            raise ValueError(
                f"""Length of item_counts and embeddings must be the same.
        len(item_counts)={len(item_counts)}, embeddings.shape={embeddings.shape}
        """
            )
        cluster_assignment: List[int] = [-1] * embeddings.shape[0]
        cluster_info_list: List[GraphClustering.ClusterInfo] = []
        for i in range(len(clusters)):
            total_size: int = 0
            distinct_size: int = len(clusters[i])
            for j in clusters[i]:
                cluster_assignment[j] = i
                if item_counts is not None:
                    total_size += item_counts[j]
            if item_counts is None:
                total_size = distinct_size
            cluster_info_list.append(
                self.ClusterInfo(
                    cluster_id=i,
                    distinct_size=distinct_size,
                    total_size=total_size,
                )
            )
        # Re-index cluster ids by total size in descending order
        # Largest cluster by total size will now be cluster id=0
        cluster_info_list.sort(
            key=lambda x: (x.total_size, x.distinct_size), reverse=True
        )
        ca: np.ndarray = np.array(cluster_assignment)
        final_ca: np.ndarray = np.full(ca.shape, -1)
        for i in range(len(cluster_info_list)):
            prev: int = cluster_info_list[i].cluster_id
            final_ca[ca == prev] = i
            cluster_info_list[i] = cluster_info_list[i]._replace(cluster_id=i)
        return GraphClustering.Result(
            cluster_assignment=final_ca.tolist(),
            clusters=cluster_info_list,
        )


class GreedyModularityCommunities(GraphClustering):
    def __init__(self, resolution: float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution

    def __call__(
        self,
        embeddings: np.ndarray,
        score_cutoff: float,
        item_counts: Optional[List[int]] = None,
        *args,
        **kwargs,
    ) -> GraphClustering.Result:
        # A list of frozensets of nodes, one for each community. Sorted by length with largest communities first.
        clusters = nx.community.greedy_modularity_communities(
            self._graph(embeddings=embeddings, score_cutoff=score_cutoff),
            weight=self.edge_attr,
            resolution=self.resolution,
        )
        return self._postprocess(
            clusters=clusters, embeddings=embeddings, item_counts=item_counts
        )


class AsyncLpaCommunities(GraphClustering):
    def __init__(self, seed: int = 31, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def __call__(
        self,
        embeddings: np.ndarray,
        score_cutoff: float,
        item_counts: Optional[List[int]] = None,
        *args,
        **kwargs,
    ) -> GraphClustering.Result:
        # A list of frozensets of nodes, one for each community. Sorted by length with largest communities first.
        clusters = nx.community.asyn_lpa_communities(
            self._graph(embeddings=embeddings, score_cutoff=score_cutoff),
            weight=self.edge_attr,
            seed=self.seed,
        )
        # `asyn_lpa_communities` returns a generator, so we convert that to a Sequence.
        return self._postprocess(
            clusters=list(clusters), embeddings=embeddings, item_counts=item_counts
        )
