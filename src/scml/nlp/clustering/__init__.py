from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["TfIdfClustering", "Keyword", "keywords"]


class Keyword(NamedTuple):
    score: float
    text: str


def keywords(
    docs: Iterable[str],
    stop_words: Optional[List[str]] = None,
    analyzer: str = "word",
    lowercase: bool = True,
    ngram_range: Tuple[int, int] = (1, 1),
    max_df: Union[float, int] = 1.0,
    min_df: Union[float, int] = 1,
    max_features: Optional[int] = None,
    vocabulary: Optional[Union[Dict, Iterable[str]]] = None,
) -> Iterator[Keyword]:
    vectorizer = TfidfVectorizer(
        lowercase=lowercase,
        analyzer=analyzer,
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        vocabulary=vocabulary,
        dtype=np.float32,  # vectorizer does not support float16
    )
    # rank tf-idf weights across all docs
    x = vectorizer.fit_transform(docs).toarray().flatten()
    ngrams = vectorizer.get_feature_names_out()
    ordering = (-x).argsort()
    seen: Set[str] = set()
    i = 0
    while len(seen) < len(ngrams) and i < len(ordering):
        j = ordering[i] % len(ngrams)
        if ngrams[j] not in seen:
            seen.add(ngrams[j])
            yield Keyword(score=x[ordering[i]], text=ngrams[j])
        i += 1


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
        dtype=np.float16,
    ):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            analyzer=analyzer,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            dtype=np.float32,  # vectorizer does not support float16
        )
        X = self.vectorizer.fit_transform(docs).toarray()
        n = X.shape[0]
        self.sim = np.full((n, n), 1, dtype=dtype)
        for i in range(n):
            for j in range(i + 1, n):
                # vectors are already L2 norm, so just take dot product
                cos_sim: float = float(np.dot(X[i], X[j]))
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
        res = [-1] * len(self.docs)
        for i in range(len(cs)):
            for j in cs[i]:
                res[j] = i
        return res
