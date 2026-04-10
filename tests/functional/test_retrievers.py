"""
Functional tests for retrieve_dspy retrievers.

These tests run against a live Weaviate instance and require:
  - WEAVIATE_URL and WEAVIATE_API_KEY env vars
  - OPENAI_API_KEY env var (for LLM-based retrievers)
  - VOYAGE_API_KEY env var (for CrossEncoderReranker with Voyage)
  - A populated collection named "BrightBiology_Default" with a "content" property

Run with:
    pytest tests/functional/test_retrievers.py -v -s
"""

import os

import pytest
import voyageai
import weaviate

from retrieve_dspy.models import DSPyAgentRAGResponse, ObjectFromDB, RerankerClient
from retrieve_dspy.retrievers import (
    BaseRetriever,
    CrossEncoderReranker,
    HyDE_QueryExpander,
    MultiQueryWriter,
    QUIPLER,
    SearchQueryWriter,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

COLLECTION = "BrightBiology_Default"
PROPERTY = "content"
TEST_QUERY = (
    "Why are fearful stimuli more powerful at night? "
    "For example, horror movies appear to be scarier when viewed at night "
    "than during broad day light."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def weaviate_client():
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(os.environ["WEAVIATE_API_KEY"]),
    )
    yield client
    client.close()


@pytest.fixture(scope="session")
def voyage_client():
    return voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_valid_response(response, min_sources=1):
    """Common assertions shared across all retriever tests."""
    assert isinstance(response, DSPyAgentRAGResponse)
    assert len(response.sources) >= min_sources, (
        f"Expected at least {min_sources} sources, got {len(response.sources)}"
    )
    for src in response.sources:
        assert isinstance(src, ObjectFromDB)
        assert src.object_id
        assert src.content
    assert isinstance(response.searches, list)
    assert len(response.searches) >= 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaseRetriever:
    """Raw search with no LLM — just Weaviate hybrid/bm25/vector."""

    def test_hybrid_search(self, weaviate_client):
        retriever = BaseRetriever(
            collection_name=COLLECTION,
            weaviate_client=weaviate_client,
            target_property_name=PROPERTY,
            retrieved_k=10,
            search_type="hybrid",
            verbose=False,
        )
        response = retriever.forward(TEST_QUERY, weaviate_client=weaviate_client)
        assert_valid_response(response, min_sources=10)

    def test_bm25_search(self, weaviate_client):
        retriever = BaseRetriever(
            collection_name=COLLECTION,
            weaviate_client=weaviate_client,
            target_property_name=PROPERTY,
            retrieved_k=10,
            search_type="keyword",
            verbose=False,
        )
        response = retriever.forward(TEST_QUERY, weaviate_client=weaviate_client)
        assert_valid_response(response, min_sources=1)

    def test_vector_search(self, weaviate_client):
        retriever = BaseRetriever(
            collection_name=COLLECTION,
            weaviate_client=weaviate_client,
            target_property_name=PROPERTY,
            retrieved_k=10,
            search_type="vector",
            verbose=False,
        )
        response = retriever.forward(TEST_QUERY, weaviate_client=weaviate_client)
        assert_valid_response(response, min_sources=10)


class TestSearchQueryWriter:
    """Single LLM-rewritten query -> hybrid search."""

    def test_forward(self, weaviate_client):
        retriever = SearchQueryWriter(
            collection_name=COLLECTION,
            target_property_name=PROPERTY,
            retrieved_k=10,
            search_type="hybrid",
            verbose=False,
        )
        response = retriever.forward(TEST_QUERY, weaviate_client=weaviate_client)
        assert_valid_response(response)
        # The search query should differ from the original question
        assert response.searches[0] != ""


class TestMultiQueryWriter:
    """LLM generates multiple queries, results pooled."""

    def test_separate_searches(self, weaviate_client):
        retriever = MultiQueryWriter(
            collection_name=COLLECTION,
            target_property_name=PROPERTY,
            retrieved_k=5,
            search_with_queries_concatenated=False,
            verbose=False,
        )
        response = retriever.forward(TEST_QUERY)
        assert_valid_response(response)
        assert len(response.searches) > 1, "Expected multiple search queries"

    def test_concatenated_search(self, weaviate_client):
        retriever = MultiQueryWriter(
            collection_name=COLLECTION,
            target_property_name=PROPERTY,
            retrieved_k=10,
            search_with_queries_concatenated=True,
            verbose=False,
        )
        response = retriever.forward(TEST_QUERY)
        assert_valid_response(response)


class TestHyDEQueryExpander:
    """Hypothetical document generation -> search."""

    def test_forward(self, weaviate_client):
        retriever = HyDE_QueryExpander(
            collection_name=COLLECTION,
            target_property_name=PROPERTY,
            weaviate_client=weaviate_client,
            retrieved_k=10,
            search_type="hybrid",
            verbose=False,
        )
        response = retriever.forward(TEST_QUERY, weaviate_client=weaviate_client)
        assert_valid_response(response)
        # HyDE search string should be a generated passage, not the original query
        assert len(response.searches[0]) > len(TEST_QUERY)


class TestCrossEncoderReranker:
    """Retrieve + rerank with Voyage cross-encoder."""

    def test_forward(self, weaviate_client, voyage_client):
        retriever = CrossEncoderReranker(
            collection_name=COLLECTION,
            target_property_name=PROPERTY,
            weaviate_client=weaviate_client,
            retrieved_k=30,
            reranked_k=10,
            verbose=False,
        )
        response = retriever.forward(
            question=TEST_QUERY,
            weaviate_client=weaviate_client,
            reranker_clients=[RerankerClient(name="voyage", client=voyage_client)],
        )
        assert_valid_response(response, min_sources=10)
        # Scores should be populated after reranking
        assert response.sources[0].relevance_score is not None


class TestQUIPLER:
    """Full composition: multi-query -> search + CE rerank -> RRF fusion."""

    def test_forward(self, weaviate_client, voyage_client):
        retriever = QUIPLER(
            collection_name=COLLECTION,
            target_property_name=PROPERTY,
            weaviate_client=weaviate_client,
            reranker_clients=[RerankerClient(name="voyage", client=voyage_client)],
            retrieved_k=20,
            reranked_k=10,
            verbose=False,
        )
        response = retriever.forward(
            question=TEST_QUERY,
            weaviate_client=weaviate_client,
            reranker_clients=[RerankerClient(name="voyage", client=voyage_client)],
        )
        assert_valid_response(response)
        assert len(response.searches) >= 1
