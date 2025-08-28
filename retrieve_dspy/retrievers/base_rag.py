import abc
import os
from typing import Optional

import dspy

from retrieve_dspy.models import DSPyAgentRAGResponse

class BaseRAG(dspy.Module):
    def __init__(
        self, 
        collection_name: str, 
        target_property_name: Optional[str] = "content",
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True, 
        retrieved_k: Optional[int] = 5,
        verbose_signature: Optional[bool] = True,
    ) -> None:
        self.collection_name = collection_name
        self.target_property_name = target_property_name
        self.verbose = verbose
        self.search_only = search_only
        self.retrieved_k = retrieved_k
        self.verbose_signature = verbose_signature
        
        # TODO: Interface ablating `lms` here
        lm = dspy.LM(
            "openai/gpt-4.1-mini",
            cache=False, 
            api_key=os.getenv("OPENAI_API_KEY"),
        )
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

    @abc.abstractmethod
    def forward(self, question: str) -> DSPyAgentRAGResponse: ...
    
    @abc.abstractmethod
    async def aforward(self, question: str) -> DSPyAgentRAGResponse: ...