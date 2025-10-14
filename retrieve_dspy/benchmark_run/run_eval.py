import numpy as np
import yaml
import time
from pathlib import Path

import retrieve_dspy
from retrieve_dspy.metrics import create_metric
from retrieve_dspy.datasets.in_memory import in_memory_dataset_loader, prepare_random_subset
from retrieve_dspy.clients import get_weaviate_client, get_voyage_client, get_and_connect_weaviate_async_client, get_voyage_async_client

from retriever_builder import build_retriever
from retrieve_dspy.benchmark_run.eval_utils import (
    load_config,
    create_metrics_dict,
    load_dataset,
    print_trial_results,
    print_final_results,
    get_evaluator,
    offline_recall_evaluator
)

def main():
    # Load configuration
    config = load_config()

    # Build retriever from config
    print(f"Building retriever with config: {config['retriever']}")
    rag_pipeline = build_retriever(
        retriever_config=config["retriever"],
        use_async=config.get("use_async", False),
        dataset_config=config["dataset"],
        lm_config=config.get("language_models")
    )

    print(f"Successfully created {type(rag_pipeline).__name__} pipeline")
    
    # Load dataset
    queries = load_dataset(config["dataset"])
    
    # Create metrics
    metrics = create_metrics_dict(config["metrics"])
    
    # Primary metric (first one in the list, or recall@1 by default)
    primary_metric = create_metric(
        metric_type="recall",
        k=1,
        verbose=True
    )
    
    # Initialize tracking variables
    eval_config = config["evaluation"]
    num_trials = eval_config["num_trials"]
    scores = []
    offline_scores_across_trials = {metric_name: [] for metric_name in metrics.keys()}
    used_qs = None  # TODO: Leave this when introducing fine-tuned retrievers
    
    print(f"Running evaluation with {config['retriever']['type']} retriever")
    print(f"Dataset: {config['dataset']['name']} (Collection: {config['dataset']['collection_name']})")
    print(f"Trials: {num_trials}, Samples per trial: {eval_config['num_samples']}")
    print(f"Retriever config: retrieved_k={config['retriever'].get('retrieved_k', 'N/A')}")
    
    # Run evaluation trials
    for trial in range(num_trials):
        print(f"\nRunning trial {trial + 1}/{num_trials}")
        
        # Prepare test set
        testset = prepare_random_subset(
            queries=queries,
            num_samples=eval_config["num_samples"],
            seed=eval_config["seed"],
            samples_used_in_training=used_qs,
        )
        
        # Create evaluator
        evaluator = get_evaluator(
            testset=testset,
            metric=primary_metric,
        )
        
        # Run evaluation
        dspy_evaluator_kwargs = {
            "num_threads": eval_config["num_threads"]
        }
        
        evaluator_result = evaluator(rag_pipeline, **dspy_evaluator_kwargs)
        primary_score = evaluator_result.score
        scores.append(primary_score)
        all_results = evaluator_result.results
        
        # Calculate offline metrics
        print("Calculating offline metrics...")
        offline_scores = offline_recall_evaluator(
            results=all_results,
            metrics=metrics
        )
        
        # Store results
        for key, value in offline_scores.items():
            offline_scores_across_trials[key].append(value)
        
        # Print trial results
        print_trial_results(trial, num_trials, primary_score, offline_scores)
        
        # Optional sleep to avoid rate limits (uncomment if needed)
        # print("Sleeping to avoid rate limits...")
        # time.sleep(60)
    
    # Print final results
    print_final_results(scores, offline_scores_across_trials, metrics)


if __name__ == "__main__":
    main()