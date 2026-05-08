import os
from typing import Optional

import dspy
import numpy as np
from pyversity import diversify, Strategy
import weaviate

from retrieve_dspy.database.weaviate_database import (
    weaviate_search_tool,
    async_weaviate_search_tool
)
from retrieve_dspy.models import DSPyAgentRAGResponse, MultiLMConfig
from retrieve_dspy.retrievers.embeddings_registry import get_embedding_headers

class BaseRetriever(dspy.Module):
    def __init__(
        self,
        collection_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True,
        verbose_signature: Optional[bool] = True,
        multi_lm_configs: Optional[list[MultiLMConfig]] = None,
        embedding_model: Optional[str] = None,
        retrieved_k: Optional[int] = 20,
        diversity_weight: Optional[float] = 0,
        search_type: str = "hybrid",
    ) -> None:
        self.collection_name = collection_name
        self.weaviate_client = weaviate_client
        self.target_property_name = target_property_name
        self.verbose = verbose
        self.search_only = search_only
        self.verbose_signature = verbose_signature
        self.multi_lm_configs = multi_lm_configs
        self.embedding_model = embedding_model
        self.retrieved_k = retrieved_k
        self.diversity_weight = diversity_weight
        self.search_type = search_type
        if self.multi_lm_configs:
            self._multi_lm_configs_to_dict()
        else:
            self.multi_lm_configs_dict = None

        default_lm = "openai/gpt-5.4-mini"

        lm = dspy.LM(
            default_lm,
            cache=False,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        print(f"\033[95mDSPy configured with default LM: {default_lm}\033[0m")

        dspy.configure(lm=lm, track_usage=True)

    @staticmethod
    def _merge_usage(*usages: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        merged: dict[str, dict[str, int]] = {}
        for usage in usages:
            # Skip None values
            if usage is None:
                continue
            for lm_id, stats in usage.items():
                bucket = merged.setdefault(
                    lm_id, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                bucket["prompt_tokens"] += stats.get("prompt_tokens", 0)
                bucket["completion_tokens"] += stats.get("completion_tokens", 0)
        return merged

    def get_embedding_headers(self) -> dict[str, str]:
        if self.embedding_model is None:
            return {}
        return get_embedding_headers(self.embedding_model)

    def _multi_lm_configs_to_dict(self):
        self.multi_lm_configs_dict = {config.signature_name: config.lm for config in self.multi_lm_configs}

    def forward(self, question: str, weaviate_client: Optional[weaviate.WeaviateClient] = None) -> DSPyAgentRAGResponse:
        if weaviate_client is None:
            if isinstance(self.weaviate_client, weaviate.WeaviateClient):
                weaviate_client = self.weaviate_client

        if self.diversity_weight > 0:
            retrieved_k = self.retrieved_k * 2
        else:
            retrieved_k = self.retrieved_k

        sources = weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=retrieved_k,
            weaviate_client=weaviate_client,
            return_vector=True,
            return_score=True,
            search_type=self.search_type,
        )

        if self.diversity_weight > 0:
            vectors = np.array([source.vector for source in sources])
            scores = np.array([source.relevance_score for source in sources])
            diversified_result = diversify(
                embeddings=vectors,
                scores=scores,
                k=retrieved_k,
                strategy=Strategy.MMR,
                diversity=self.diversity_weight,
            )
            diversity_ranked_indices = diversified_result.indices
            sources = [sources[i] for i in diversity_ranked_indices]

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        if not self.search_only:
            print("")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            usage={},
        )

    async def aforward(self, question: str, weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None) -> DSPyAgentRAGResponse:
        if weaviate_async_client is None:
            if isinstance(self.weaviate_async_client, weaviate.WeaviateAsyncClient):
                weaviate_async_client = self.weaviate_async_client

        if self.diversity_weight > 0:
            retrieved_k = self.retrieved_k * 2
        else:
            retrieved_k = self.retrieved_k

        sources = await async_weaviate_search_tool(
            query=question,
            collection_name=self.collection_name,
            target_property_name=self.target_property_name,
            retrieved_k=retrieved_k,
            weaviate_async_client=weaviate_async_client,
            return_vector=True,
            return_score=True,
            search_type=self.search_type,
        )

        if self.diversity_weight > 0:
            vectors = np.array([source.vector for source in sources])
            scores = np.array([source.relevance_score for source in sources])
            diversified_result = diversify(
                embeddings=vectors,
                scores=scores,
                k=retrieved_k,
                strategy=Strategy.MMR,
                diversity=self.diversity_weight,
            )
            diversity_ranked_indices = diversified_result.indices
            sources = [sources[i] for i in diversity_ranked_indices]

        if self.verbose:
            print(f"\033[96m Returning {len(sources)} Sources!\033[0m")

        if not self.search_only:
            print("")

        return DSPyAgentRAGResponse(
            final_answer="",
            sources=sources,
            searches=[question],
            usage={},
        )
