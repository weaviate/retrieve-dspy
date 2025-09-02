import numpy as np
import time

import retrieve_dspy
from retrieve_dspy.metrics import create_metric
from retrieve_dspy.datasets.in_memory import in_memory_dataset_loader, prepare_random_subset
from retrieve_dspy.clients import get_weaviate_client, get_voyage_client

weaviate_client = get_weaviate_client()
voyage_client = get_voyage_client()

'''
rag_pipeline = retrieve_dspy.VanillaRAG(
    weaviate_client=weaviate_client,
    collection_name="EnronEmails",
    target_property_name="email_body",
    verbose=False,
)
'''

'''
rag_pipeline = retrieve_dspy.RAGFusion(
    weaviate_client=weaviate_client,
    collection_name="EnronEmails",
    target_property_name="email_body",
    retrieved_k=50,
    reranked_k=20,
    verbose=True,
    verbose_signature=True
)
'''

'''
rag_pipeline = retrieve_dspy.CrossEncoderReranker(
    weaviate_client=weaviate_client,
    reranker_clients=[voyage_client],
    collection_name="EnronEmails",
    target_property_name="email_body",
    retrieved_k=50,
    reranked_k=20,
    reranker_provider="voyage",
    verbose=True
)
'''

'''
rag_pipeline = retrieve_dspy.LayeredReranker(
    weaviate_client=weaviate_client,
    reranker_clients=[voyage_client],
    collection_name="EnronEmails",
    target_property_name="email_body",
    return_property_name="email_body",
    retrieved_k=50,
    reranked_N=20,
    reranked_M=5,
    reranker_provider="voyage",
    listwise_reranker_strategy="BestMatch",
    verbose=True
)
'''

'''
rag_pipeline = retrieve_dspy.LayeredReranker(
    weaviate_client=weaviate_client,
    reranker_clients=[voyage_client],
    collection_name="WixKB",
    target_property_name="contents",
    return_property_name="contents",
    retrieved_k=50,
    reranked_N=20,
    reranked_M=5,
    reranker_provider="voyage",
    listwise_reranker_strategy="BestMatch",
    verbose=True
)
'''

rag_pipeline = retrieve_dspy.VanillaRAG(
    weaviate_client=weaviate_client,
    collection_name="EnronEmails",
    target_property_name="email_body",
    verbose=True
)


#print(rag_pipeline.__class__.__name__)

#rag_pipeline.load("./optimization_runs/2_gepa_optimized_query_expander.json")
#used_qs = retrieve_dspy.utils.load_training_questions("./optimization_runs/2_gepa_query_expander_training_samples.jsonl")
used_qs = None

#print(f"\033[92m{rag_pipeline.expand_query.signature}\033[0m")

NUM_TRIALS = 1
scores = []

metric = create_metric(
    metric_type="recall",
    k=20,
    verbose=False
)

recall_metrics = {
    'recall@1': create_metric(
        metric_type="recall",
        k=1,
        verbose=False
    ),
    'recall@5': create_metric(
        metric_type="recall",
        k=5,
        verbose=False
    ),
    'recall@20': create_metric(
        metric_type="recall",
        k=20,
        verbose=False
    )
}

offline_scores_across_trials = {metric_name: [] for metric_name in recall_metrics.keys()}

# queries = load_queries_in_memory(dataset_name="enron")
# testset = prepare_random_subset(
#   queries,
#   num_samples=50,
#   seed=0,
#   samples_used_in_training=used_qs,
# )

_, queries = in_memory_dataset_loader(dataset_name="enron")

for trial in range(NUM_TRIALS):
    print(f"\nRunning trial {trial + 1}/{NUM_TRIALS}")

    testset = prepare_random_subset(
        queries=queries,
        num_samples=150, #150
        seed=42,
        samples_used_in_training=used_qs,
    )

    evaluator = retrieve_dspy.utils.get_evaluator(
        testset=testset,
        metric=metric,
    )

    dspy_evaluator_kwargs = {
        "num_threads": 1
    }

    evaluator_result = evaluator(rag_pipeline, **dspy_evaluator_kwargs)
    score = evaluator_result.score
    scores.append(score)
    all_results = evaluator_result.results
    print("Running eval for all metrics...")
    offline_scores = retrieve_dspy.utils.offline_recall_evaluator(
        results=all_results,
        metrics=recall_metrics
    )
    
    for key, value in offline_scores.items():
        print(f"\033[96m{key}\033[0m: \033[92m{value:.3f}\033[0m")
        offline_scores_across_trials[key].append(value)

    #print("Sleeping to avoid rate limits...")
    #time.sleep(60)


print("\n" + "="*60)
print("ORIGINAL METRIC RESULTS ACROSS TRIALS:")
print("="*60)
scores = np.array(scores)
print(f"Individual scores: {[f'{score:.3f}' for score in scores]}")
print(f"Min score: {scores.min():.3f}")
print(f"Max score: {scores.max():.3f}") 
print(f"\033[92mMean score: {scores.mean():.3f}\033[0m")
print(f"Std dev: {scores.std():.3f}")

print("\n" + "="*60)
print("OFFLINE METRICS RESULTS ACROSS TRIALS:")
print("="*60)

for metric_name in recall_metrics.keys():
    metric_scores = np.array(offline_scores_across_trials[metric_name])
    print(f"\n\033[96m{metric_name}:\033[0m")
    print(f"  Individual scores: {[f'{score:.3f}' for score in metric_scores]}")
    print(f"  Min score:  {metric_scores.min():.3f}")
    print(f"  Max score:  {metric_scores.max():.3f}")
    print(f"  \033[92mMean score: {metric_scores.mean():.3f}\033[0m")
    print(f"  Std dev:    {metric_scores.std():.3f}")