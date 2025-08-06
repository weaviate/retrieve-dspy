import numpy as np

import retrieve_dspy
from retrieve_dspy.metrics import create_metric
from retrieve_dspy.datasets.in_memory import load_queries_in_memory

'''
rag_pipeline = retrieve_dspy.ListwiseReranker(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    diverse_ranker=True,
    retrieved_k=50,
    reranked_k=20
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

rag_pipeline = retrieve_dspy.VanillaRAG(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=100,
    verbose=True
)

rag_pipeline = retrieve_dspy.CrossEncoderReranker(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=50,
    reranked_k=20,
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
'''

rag_pipeline = retrieve_dspy.QueryExpander(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=20,
    verbose=True
)


'''
rag_pipeline = retrieve_dspy.LayeredReranker(
    collection_name="FreshstackAngular",
    target_property_name="docs_text",
    retrieved_k=100,
    reranked_N=50,
    reranked_M=20,
    verbose=True
)
'''

# rag_pipeline.load("./notebooks/mipro_optimizer_query_writer.json")

NUM_TRIALS = 5
scores = []

for trial in range(NUM_TRIALS):
    print(f"\nRunning trial {trial + 1}/{NUM_TRIALS}")
    
    trainset, testset = load_queries_in_memory(
        dataset_name="freshstack-angular",
        train_samples=20,
        test_samples=20
    )

    metric = create_metric(
        metric_type="coverage",
        dataset_name="freshstack-angular"
    )

    evaluator = retrieve_dspy.utils.get_evaluator(
        testset=testset,
        metric=metric,
    )

    dspy_evaluator_kwargs = {
        "num_threads": 4
    }

    score = evaluator(rag_pipeline, **dspy_evaluator_kwargs)
    scores.append(score)

scores = np.array(scores)
print("\nResults across trials:")
print(f"Min score: {scores.min():.3f}")
print(f"Max score: {scores.max():.3f}") 
print(f"Mean score: {scores.mean():.3f}")
print(f"Std dev: {scores.std():.3f}")