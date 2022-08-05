from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["TfIdfClustering"]


class TfIdfClustering:
    def __init__(
        self,
        docs: Sequence[str],
        similarity_min: float = 0,
        stop_words: Optional[List[str]] = None,
        analyzer: str = "word",
        lowercase: bool = True,
        ngram_range: Tuple[int, int] = (1, 1),
        max_df: Union[float, int] = 1.0,
        min_df: Union[float, int] = 1,
        max_features: Optional[int] = None,
        vocabulary: Optional[Union[Dict, Iterable[str]]] = None,
        dtype=np.float32,
    ):
        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            analyzer=analyzer,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            dtype=dtype,
        )
        self.X = self.vectorizer.fit_transform(docs).toarray()
        n = self.X.shape[0]
        self.sim = np.full((n, n), 1)
        for i in range(n):
            for j in range(i + 1, n):
                # vectors are already L2 norm, so just take dot product
                cos_sim: float = float(np.dot(self.X[i], self.X[j]))
                # prune edges below threshold
                if cos_sim < similarity_min:
                    cos_sim = 0
                self.sim[i][j] = cos_sim
                self.sim[j][i] = cos_sim
        # undirected weighted graph
        self.G = nx.from_numpy_array(self.sim, parallel_edges=False)

    def greedy_modularity_communities(
        self,
        resolution: float = 1,
    ) -> List[int]:
        cs = nx.community.greedy_modularity_communities(
            self.G,
            resolution=resolution,
        )
        res = [-1] * self.X.shape[0]
        for i in range(len(cs)):
            for j in cs[i]:
                res[j] = i
        return res
