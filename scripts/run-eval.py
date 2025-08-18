import numpy as np

import retrieve_dspy
from retrieve_dspy.metrics import create_metric
from retrieve_dspy.datasets.in_memory import load_queries_in_memory

'''
rag_pipeline = retrieve_dspy.CrossEncoderReranker(
    collection_name="EnronEmails",
    target_property_name="email_body",
    retrieved_k=50,
    reranked_k=20,
    reranker_provider="voyage",
    verbose=True
)

rag_pipeline = retrieve_dspy.ListwiseReranker(
    collection_name="EnronEmails",
    target_property_name="email_body_vector",
    return_property_name="email_summary",
    retrieved_k=5,
    reranked_k=5,
    verbose=True
)

rag_pipeline = retrieve_dspy.VanillaRAG(
    collection_name="EnronEmails",
    target_property_name="email_body_vector",
    retrieved_k=5,
    verbose=True
)

rag_pipeline = retrieve_dspy.SummarizedListwiseReranker(
    collection_name="EnronEmails",
    target_property_name="email_body_vector",
    return_property_name="email_body",
    retrieved_k=5,
    reranked_k=5,
    verbose=True
)
'''

rag_pipeline = retrieve_dspy.CrossEncoderReranker(
    collection_name="EnronEmails",
    target_property_name="email_body_vector",
    return_property_name="email_body",
    reranker_provider="hybrid",
    retrieved_k=50,
    reranked_k=20,
    verbose=True
)

#print(rag_pipeline.__class__.__name__)

#rag_pipeline.load("./optimization_runs/2_gepa_optimized_query_expander.json")
#used_qs = retrieve_dspy.utils.load_training_questions("./optimization_runs/2_gepa_query_expander_training_samples.jsonl")
used_qs = None

#print(f"\033[92m{rag_pipeline.expand_query.signature}\033[0m")

NUM_TRIALS = 3
scores = []

metric = create_metric(
    metric_type="recall",
    dataset_name="enron",
    k=1
)

recall_metrics = {
    'recall@1': create_metric(
        metric_type="recall",
        dataset_name="enron",
        k=1,
        verbose=False
    ),
    'recall@5': create_metric(
        metric_type="recall",
        dataset_name="enron",
        k=5,
        verbose=False
    ),
    'recall@20': create_metric(
        metric_type="recall",
        dataset_name="enron",
        k=20,
        verbose=False
    )
}

offline_scores_across_trials = {metric_name: [] for metric_name in recall_metrics.keys()}

for trial in range(NUM_TRIALS):
    print(f"\nRunning trial {trial + 1}/{NUM_TRIALS}")

    trainset, testset = load_queries_in_memory(
        dataset_name="enron",
        train_samples=20,
        test_samples=20,
        training_samples=used_qs,
        seed=trial
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