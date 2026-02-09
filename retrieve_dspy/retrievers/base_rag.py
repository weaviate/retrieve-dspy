import abc
import os
from typing import Optional

import dspy
import weaviate

from retrieve_dspy.models import DSPyAgentRAGResponse, MultiLMConfig

class BaseRAG(dspy.Module):
    def __init__(
        self,
        collection_name: str, 
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True, 
        verbose_signature: Optional[bool] = True,
        multi_lm_configs: Optional[list[MultiLMConfig]] = None,
    ) -> None:
        self.collection_name = collection_name
        self.weaviate_client = weaviate_client
        self.target_property_name = target_property_name
        self.verbose = verbose
        self.search_only = search_only
        self.verbose_signature = verbose_signature
        self.multi_lm_configs = multi_lm_configs
        if self.multi_lm_configs:
            self._multi_lm_configs_to_dict()
        else:
            self.multi_lm_configs_dict = None

        default_lm = "openai/gpt-4.1-mini"

        lm = dspy.LM(
            default_lm,
            temperature=1.0,
            max_tokens=32000,
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

    def _multi_lm_configs_to_dict(self):
        self.multi_lm_configs_dict = {config.signature_name: config.lm for config in self.multi_lm_configs}

    @abc.abstractmethod
    def forward(self, question: str, weaviate_client: Optional[weaviate.WeaviateClient] = None) -> DSPyAgentRAGResponse: ...
    
    @abc.abstractmethod
    async def aforward(self, question: str, weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None) -> DSPyAgentRAGResponse: ...