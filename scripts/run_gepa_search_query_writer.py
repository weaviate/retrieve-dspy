"""
Run GEPA optimization on SearchQueryWriter with BM25 search
using the ReasonIR BRIGHT Biology dataset.
"""

from retrieve_dspy import SearchQueryWriter
from retrieve_dspy.optimize import run_gepa

import dspy

print (dspy.__version__)

retriever = SearchQueryWriter(
    collection_name="BrightBiology_Default",
    target_property_name="content",
    retrieved_k=5,
    verbose=False,
    search_type="bm25",
)

optimized = run_gepa(
    retriever=retriever,
    auto="light",
    use_wandb=True,
    wandb_init_kwargs={
        "project": "retrieve-dspy-gepa",
        "name": "search-query-writer-bm25-reasonir-biology",
    },
)
