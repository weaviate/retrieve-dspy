import numpy as np

import retrieve_dspy
from retrieve_dspy.metrics import create_metric
from retrieve_dspy.datasets.in_memory import load_queries_in_memory

'''
rag_pipeline = retrieve_dspy.VanillaRAG(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=100,
    verbose=True
)

rag_pipeline = retrieve_dspy.QueryWriterWithListwiseReranker(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=10,
    reranked_k=20
)

rag_pipeline = retrieve_dspy.MultiQueryWriter(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=100,
    search_with_queries_concatenated=True
)

rag_pipeline = retrieve_dspy.CrossEncoderReranker(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=50,
    reranked_k=20,
    verbose=True
)

rag_pipeline = retrieve_dspy.ListwiseReranker(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=50,
    reranked_k=10,
    diverse_ranker=True,
    verbose=True
)

rag_pipeline = retrieve_dspy.MultiQueryWriterWithCrossEncoderReranker(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=50,
    reranked_k=20,
    search_with_queries_concatenated=False,
    two_stage_reranking=True,
    per_query_top_k=20,
    verbose=True
)

rag_pipeline = retrieve_dspy.LoopingQueryWriter(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=10,
    max_loops=1,
    verbose=True
)

rag_pipeline = retrieve_dspy.QueryExpander(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=50,
    verbose=True
)

rag_pipeline = retrieve_dspy.LayeredReranker(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=100,
    reranked_N=50,
    reranked_M=20,
    verbose=True
)

rag_pipeline = retrieve_dspy.DecomposeAndExpand(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=20,
    verbose=True
)

rag_pipeline = retrieve_dspy.CrossEncoderReranker(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=100,
    reranked_k=50,
    summarize_query=False,
    verbose=True
)

rag_pipeline = retrieve_dspy.VanillaRAG(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=20,
    verbose=True
)
'''
'''
rag_pipeline = retrieve_dspy.QueryExpanderWithHint(
    collection_name="FreshstackLangchain",
    target_property_name="docs_text",
    retrieved_k=10,
    verbose=True
)
'''

rag_pipeline = retrieve_dspy.VanillaRAG(
    collection_name="EnronEmails",
    target_property_name="email_body",
    retrieved_k=20,
    verbose=True
)

#print(rag_pipeline.__class__.__name__)

#rag_pipeline.load("./optimization_runs/2_gepa_optimized_query_expander.json")
#used_qs = retrieve_dspy.utils.load_training_questions("./optimization_runs/2_gepa_query_expander_training_samples.jsonl")
used_qs = None

#print(f"\033[92m{rag_pipeline.expand_query.signature}\033[0m")

NUM_TRIALS = 5
scores = []

for trial in range(NUM_TRIALS):
    print(f"\nRunning trial {trial + 1}/{NUM_TRIALS}")

    trainset, testset = load_queries_in_memory(
        dataset_name="enron",
        train_samples=20,
        test_samples=20,
        training_samples=used_qs,
        seed=trial,
    )

    metric = create_metric(
        metric_type="recall",
        dataset_name="enron"
    )

    evaluator = retrieve_dspy.utils.get_evaluator(
        testset=testset,
        metric=metric,
    )

    dspy_evaluator_kwargs = {
        "num_threads": 4
    }

    score = evaluator(rag_pipeline, **dspy_evaluator_kwargs).score
    scores.append(score)

scores = np.array(scores)
print(scores)

print("\n\nDEBUGGING\n\n")

print("\nResults across trials:")
print(f"Individual scores: {[f'{score:.3f}' for score in scores]}")
print(f"Min score: {scores.min():.3f}")
print(f"Max score: {scores.max():.3f}") 
print(f"Mean score: {scores.mean():.3f}")
print(f"Std dev: {scores.std():.3f}")