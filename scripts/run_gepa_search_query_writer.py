"""
Run GEPA optimization on SearchQueryWriter with BM25 search
using the ReasonIR BRIGHT Biology dataset.
"""

from retrieve_dspy import SearchQueryWriter
from retrieve_dspy.optimize import run_gepa, load_search_dataset

import dspy

print (dspy.__version__)

retriever = SearchQueryWriter(
    collection_name="BrightBiology_Default",
    target_property_name="content",
    retrieved_k=10,
    verbose=False,
    search_type="bm25",
)

trainset, testset = load_search_dataset(
    train_path="scripts/reasonir_bright_biology_train.json",
    test_path="scripts/reasonir_bright_biology_test.json",
)
print(f"Train: {len(trainset)} examples, Test: {len(testset)} examples")

optimized_retriever = run_gepa(
    retriever=retriever,
    dataset=(trainset, testset),
    auto="medium",
    train_size=50,
    val_size=30,
    use_wandb=True,
    wandb_init_kwargs={
        "project": "retrieve-dspy-gepa",
        "name": "search-query-writer-bm25-reasonir-biology",
    },
)

optimized_retriever.save("optimized-programs/search_query_writer_bm25.json")
