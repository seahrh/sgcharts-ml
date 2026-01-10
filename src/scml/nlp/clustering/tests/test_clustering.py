import math
from typing import List

import networkx as nx
import numpy as np
import pytest

import scml
from scml.nlp.clustering import *

scml.seed_everything()


@pytest.fixture
def docs2() -> List[str]:
    return [
        "10 11 12 13",
        "13 12 11 10",
        "12 11 10 13",  # this doc and above are duplicates
        "10 20 21 22",
        "10 11 20 21",
        "20 21 22 23",
        "99",
    ]


@pytest.fixture
def docs() -> List[str]:
    return [
        "apple banana apple",
        "banana orange banana",
        "orange apple banana",
    ]


def collect_keywords(*args, **kwargs):
    """Helper to exhaust the iterator."""
    return list(keywords(*args, **kwargs))


class TestKeywordsBasic:
    def test_returns_keyword_objects(self, docs):
        result = collect_keywords(docs)

        assert result
        assert all(isinstance(k, Keyword) for k in result)
        assert all(isinstance(k.score, float) for k in result)
        assert all(isinstance(k.text, str) for k in result)

    def test_keywords_are_unique(self, docs):
        result = collect_keywords(docs)
        texts = [k.text for k in result]

        assert len(texts) == len(set(texts))


class TestKeywordsRanking:
    def test_scores_are_non_increasing(self, docs):
        result = collect_keywords(docs)
        scores = [k.score for k in result]

        for a, b in zip(scores, scores[1:]):
            assert a >= b

    def test_highest_scoring_term_first(self, docs):
        result = collect_keywords(docs)

        assert result[0].text in {"apple", "banana", "orange"}
        assert result[0].score == max(k.score for k in result)


class TestKeywordsOptions:
    def test_ngram_range(self, docs):
        result = collect_keywords(docs, ngram_range=(2, 2))
        texts = [k.text for k in result]

        assert all(" " in text for text in texts)

    def test_stop_words_removed(self, docs):
        result = collect_keywords(docs, stop_words=["banana"])
        texts = [k.text for k in result]

        assert "banana" not in texts

    def test_lowercase_false(self):
        docs = ["Apple apple"]
        result = collect_keywords(docs, lowercase=False)
        texts = [k.text for k in result]

        assert "Apple" in texts
        assert "apple" in texts

    def test_vocabulary_restriction(self, docs):
        vocab = ["apple"]
        result = collect_keywords(docs, vocabulary=vocab)

        assert result == [Keyword(score=result[0].score, text="apple")]


class TestKeywordsDeterminism:
    def test_deterministic_output(self, docs):
        result1 = collect_keywords(docs)
        result2 = collect_keywords(docs)

        assert result1 == result2


class TestKeywordsNumericalStability:
    def test_scores_are_finite(self, docs):
        result = collect_keywords(docs)

        for kw in result:
            assert math.isfinite(kw.score)


class TestTfIdfClusteringGraph:
    def test_similarity_matrix_shape(self, docs):
        clu = TfIdfClustering(docs)

        n = len(docs)
        assert clu.sim.shape == (n, n)

    def test_self_similarity_is_one(self, docs):
        clu = TfIdfClustering(docs)

        assert np.allclose(np.diag(clu.sim), 1.0)

    def test_similarity_matrix_is_symmetric(self, docs):
        clu = TfIdfClustering(docs)

        assert np.allclose(clu.sim, clu.sim.T)

    def test_graph_node_count(self, docs):
        clu = TfIdfClustering(docs)

        assert clu.G.number_of_nodes() == len(docs)

    def test_graph_has_self_loops(self, docs):
        clu = TfIdfClustering(docs)

        self_loops = list(nx.selfloop_edges(clu.G))
        assert len(self_loops) == len(docs)


class TestTfIdfClusteringPruning:
    def test_no_pruning_keeps_nonzero_edges(self, docs):
        clu = TfIdfClustering(docs, similarity_min=0)

        # at least one off-diagonal edge should be nonzero
        off_diag = [
            clu.sim[i, j] for i in range(len(docs)) for j in range(len(docs)) if i != j
        ]
        assert any(v > 0 for v in off_diag)

    def test_pruning_removes_weak_edges(self, docs):
        clu = TfIdfClustering(docs, similarity_min=0.9)

        off_diag = [
            clu.sim[i, j] for i in range(len(docs)) for j in range(len(docs)) if i != j
        ]
        assert all(v == 0 or v >= 0.9 for v in off_diag)


class TestTfIdfClusteringCommunities:
    def test_asyn_lpa_output_shape(self, docs):
        clu = TfIdfClustering(docs)
        labels = clu.asyn_lpa_communities(seed=42)

        assert len(labels) == len(docs)
        assert all(isinstance(x, int) for x in labels)
        assert min(labels) >= 0

    def test_greedy_modularity_output_shape(self, docs):
        clu = TfIdfClustering(docs)
        labels = clu.greedy_modularity_communities()

        assert len(labels) == len(docs)
        assert all(isinstance(x, int) for x in labels)
        assert min(labels) >= 0

    def test_resolution_affects_granularity(self, docs):
        clu = TfIdfClustering(docs)

        coarse = clu.greedy_modularity_communities(resolution=0.5)
        fine = clu.greedy_modularity_communities(resolution=2.0)

        # lower resolution → fewer communities
        assert len(set(coarse)) <= len(set(fine))


class TestTfIdfClusteringDeterminism:
    def test_asyn_lpa_seeded_is_deterministic(self, docs):
        clu = TfIdfClustering(docs)

        labels1 = clu.asyn_lpa_communities(seed=123)
        labels2 = clu.asyn_lpa_communities(seed=123)

        assert labels1 == labels2
