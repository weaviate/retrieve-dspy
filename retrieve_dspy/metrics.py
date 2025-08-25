import asyncio
import os
from typing import Callable

from dspy import Example, Prediction
import weaviate
from weaviate.collections.classes.filters import Filter

def get_collection(weaviate_client, dataset_name: str):
    """Get the appropriate Weaviate collection for a dataset."""
    if dataset_name == "enron":
        return weaviate_client.collections.get("EnronEmails")
    elif dataset_name == "wixqa":
        return weaviate_client.collections.get("WixKB")
    elif dataset_name.startswith("freshstack-"):
        subset = dataset_name.split("-")[1].capitalize()
        return weaviate_client.collections.get(f"Freshstack{subset}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def calculate_recall_at_k(
    target_ids: list[str],
    retrieved_ids: list[str],
    k: int,
    verbose: bool = True
):
    """Calculate traditional recall@k for retrieved documents.
    
    Args:
        target_ids: List of target document IDs (ground truth).
        retrieved_ids: List of retrieved document IDs.
        k: The number of top results to consider for recall calculation.
        
    Returns:
        float: Recall@k score (0.0 to 1.0) - proportion of relevant docs
               found in the top k retrieved results.
    """
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    
    # Use sets for efficient lookup
    target_id_set = {str(id) for id in target_ids}
    retrieved_ids = [str(id) for id in retrieved_ids] if retrieved_ids else []
    
    # Consider only the top k retrieved IDs
    retrieved_ids_at_k = retrieved_ids[:k]
    
    if verbose:
        print(f"\033[96mTarget IDs: {target_id_set}\033[0m")
    
    # Find the number of relevant documents found in the top k
    found_count = sum(1 for retrieved_id in retrieved_ids_at_k if retrieved_id in target_id_set)
    
    if found_count > 0:
        if verbose:
            print(f"\033[92mRetrieved IDs @{k}: {retrieved_ids_at_k}\033[0m")
    else:
        if verbose:
            print(f"\033[91mRetrieved IDs @{k}: {retrieved_ids_at_k}\033[0m")
    
    recall = found_count / len(target_id_set) if target_id_set else 0
    if verbose:
        print(f"\033[96mRecall@{k}: {found_count}/{len(target_id_set)} = {recall:.2f}\033[0m")
    
    return recall

def calculate_coverage(retrieved_ids: list[str], nugget_data: list[dict], k: int = 1000):
    """Calculate Coverage@k metric from FreshStack.
    
    Measures the proportion of nuggets covered by the top-k retrieved documents.
    
    Args:
        retrieved_ids: List of retrieved document IDs in ranked order
        nugget_data: List of nugget information, each with 'relevant_corpus_ids' field
        k: Number of top documents to consider (default: 20)
    
    Returns:
        float: Coverage@k score (0.0 to 1.0) - proportion of nuggets covered
    """
    if not nugget_data:
        return 0.0
    
    # Convert to strings for consistent comparison
    retrieved_ids = [str(id) for id in retrieved_ids[:k]] if retrieved_ids else []
    
    covered_nuggets = set()
    nugget_coverage_details = []
    
    for i, nugget in enumerate(nugget_data):
        nugget_id = nugget.get('id', f'nugget_{i}')
        nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
        
        # Check if any relevant doc for this nugget is in top-k retrieved
        covered = any(doc_id in retrieved_ids for doc_id in nugget_relevant_ids)
        
        if covered:
            covered_nuggets.add(nugget_id)
            nugget_coverage_details.append(f"\033[92mNugget {i+1}: Covered\033[0m")
        else:
            nugget_coverage_details.append(f"\033[91mNugget {i+1}: Not covered\033[0m")
    
    coverage_score = len(covered_nuggets) / len(nugget_data)
    
    # Print summary
    print(f"\033[96mCoverage@{k} evaluation:\033[0m")
    print(f"Total nuggets: {len(nugget_data)}")
    print(f"Covered nuggets: {len(covered_nuggets)}")
    for detail in nugget_coverage_details[:5]:  # Show first 5 for brevity
        print(detail)
    if len(nugget_coverage_details) > 5:
        print(f"... and {len(nugget_coverage_details) - 5} more nuggets")
    print(f"\033[96mCoverage@{k}: {len(covered_nuggets)}/{len(nugget_data)} = {coverage_score:.2f}\033[0m")
    
    return coverage_score

def create_recall_metric(weaviate_client, dataset_name: str, k: int, verbose: bool = True) -> Callable:
    """
    Create a recall metric function that wraps the existing calculate_recall function.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        weight: Weight to apply to the recall score
        
    Returns:
        Function that calculates recall score for a single example
    """
    collection = get_collection(weaviate_client, dataset_name)
    
    def recall_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract sources from prediction
            if hasattr(prediction, 'sources') and prediction.sources:
                retrieved_ids = qa_source_parser(prediction.sources, collection)
            else:
                retrieved_ids = []
            
            # Get target IDs from example
            target_ids = example.dataset_ids
            
            # Use the existing calculate_recall function
            recall_score = calculate_recall_at_k(
                target_ids=target_ids,
                retrieved_ids=retrieved_ids,
                k=k,
                verbose=verbose
            )
            
            return recall_score
            
        except Exception as e:
            print(f"Error calculating recall: {e}")
            return 0.0
            
    return recall_metric

def create_coverage_metric(weaviate_client, dataset_name: str, k: int = 1000) -> Callable:
    """
    Create a coverage metric function that wraps the existing calculate_coverage function.
    
    Args:
        weaviate_client: Weaviate client instance
        dataset_name: Name of the dataset
        k: Number of top documents to consider (default: 100)
        
    Returns:
        Function that calculates coverage score for a single example
    """
    collection = get_collection(weaviate_client, dataset_name)
    
    def coverage_metric(example: Example, prediction, trace=None) -> float:
        try:
            # Extract sources from prediction
            if hasattr(prediction, 'sources') and prediction.sources:
                retrieved_ids = qa_source_parser(prediction.sources, collection)
            else:
                retrieved_ids = []
            
            # Get nugget data from example
            nugget_data = example.nugget_data if hasattr(example, 'nugget_data') else []
            
            # Use the existing calculate_coverage function
            coverage_score = calculate_coverage(
                retrieved_ids=retrieved_ids,
                nugget_data=nugget_data,
                k=k
            )
            
            return coverage_score
            
        except Exception as e:
            print(f"Error calculating coverage: {e}")
            return 0.0
            
    return coverage_metric

def create_metric(
    metric_type: str,
    dataset_name: str,
    **kwargs
) -> Callable:
    """
    Factory function for creating metric functions.
    
    Args:
        metric_type: Type of metric ("recall", "coverage")
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for metric configuration
        
    Returns:
        Configured metric function
    """
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    if metric_type == "recall":
        return create_recall_metric(weaviate_client, dataset_name, **kwargs)
    elif metric_type == "coverage":
        return create_coverage_metric(weaviate_client, dataset_name, **kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

def create_coverage_metric_with_feedback(
    weaviate_client: weaviate.Client,
    dataset_name: str, 
    k: int = 1000
) -> Callable:
    """
    Create a GEPA-compatible coverage metric with nuggets covered and uncovered feedback.
    """
    collection = get_collection(weaviate_client, dataset_name)
    
    def coverage_metric_with_feedback(
        example: Example, 
        prediction, 
        trace=None,
        pred_name=None,
        pred_trace=None
    ) -> Prediction:
        try:
            if hasattr(prediction, 'sources') and prediction.sources:
                retrieved_ids = qa_source_parser(prediction.sources, collection)
            else:
                retrieved_ids = []
            
            nugget_data = example.nugget_data if hasattr(example, 'nugget_data') else []
            
            retrieved_ids_str = [str(id) for id in retrieved_ids[:k]] if retrieved_ids else []
            
            covered = []
            uncovered = []
            
            for nugget in nugget_data:
                nugget_relevant_ids = [str(id) for id in nugget.get('relevant_corpus_ids', [])]
                nugget_text = nugget.get('text', '')
                
                if any(doc_id in retrieved_ids_str for doc_id in nugget_relevant_ids):
                    covered.append(nugget_text)
                else:
                    uncovered.append(nugget_text)
            
            coverage_score = len(covered) / len(nugget_data) if nugget_data else 0.0
            
            feedback = f"Nuggets covered: {covered}\nNuggets not covered: {uncovered}"
            
            return Prediction(
                score=coverage_score,
                feedback=feedback
            )
            
        except Exception as e:
            return Prediction(
                score=0.0,
                feedback=str(e)
            )
    
    return coverage_metric_with_feedback

async def main():
    # Connect to Weaviate to get some real UUIDs
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    
    # Get the collection
    collection = weaviate_client.collections.get("FreshstackLangchain")
    
    # Query for some real documents that match the nugget's relevant_corpus_ids
    # We need documents with specific dataset_ids
    target_dataset_ids = [
        'azure-openai/Basic_Samples/Embeddings/dotnet/csharp/Embedding_long_inputs.ipynb_6097_13521',
        'langchainjs/docs/core_docs/docs/how_to/query_high_cardinality.ipynb_7337_14397'
    ]
    
    # Fetch objects with these dataset_ids to get their UUIDs
    real_objects = collection.query.fetch_objects(
        filters=Filter.by_property("dataset_id").contains_any(target_dataset_ids),
        limit=2
    )
    
    gepa_metric = create_coverage_metric_with_feedback(
        dataset_name="freshstack-langchain"
    )
    
    mock_example = Example({
        'question': 'I am using the llama2 quantized model from Huggingface and loading it using ctransformers from langchain. When I run the query, I got the below warning\nNumber of tokens (512) exceeded maximum context length (512)...',
        'dataset_ids': [
            'azure-openai/Basic_Samples/Embeddings/dotnet/csharp/Embedding_long_inputs.ipynb_6097_13521',
            'azure-openai/Basic_Samples/Embeddings/dotnet/csharp/Embedding_long_inputs.ipynb_0_6096',
            'langchainjs/langchain-core/src/language_models/tests/count_tokens.test.ts_0_1089',
            'langchainjs/docs/core_docs/docs/how_to/query_high_cardinality.ipynb_7337_14397',
            'openai-cookbook/examples/data/oai_docs/fine-tuning.txt_10269_11860',
            'openai-cookbook/examples/Embedding_long_inputs.ipynb_0_7701',
            'transformers/src/transformers/generation/utils.py_66208_74709'
        ],
        'nugget_data': [
            {
                'nugget_id': '77570838_nugget_0',
                'text': 'The warning is due to the number of tokens exceeding the maximum context length.',
                'relevant_corpus_ids': [
                    'azure-openai/Basic_Samples/Embeddings/dotnet/csharp/Embedding_long_inputs.ipynb_6097_13521',
                    'azure-openai/Basic_Samples/Embeddings/dotnet/csharp/Embedding_long_inputs.ipynb_0_6096',
                    'langchainjs/langchain-core/src/language_models/tests/count_tokens.test.ts_0_1089',
                    'langchainjs/docs/core_docs/docs/how_to/query_high_cardinality.ipynb_7337_14397',
                    'openai-cookbook/examples/data/oai_docs/fine-tuning.txt_10269_11860',
                    'openai-cookbook/examples/Embedding_long_inputs.ipynb_0_7701'
                ]
            },
            {
                'nugget_id': '77570838_nugget_1',
                'text': "Adjust the 'context_length' parameter in the model configuration to a value greater than the number of tokens (e.g., 700).",
                'relevant_corpus_ids': [
                    'langchainjs/docs/core_docs/docs/how_to/query_high_cardinality.ipynb_7337_14397'
                ]
            },
            {
                'nugget_id': '77570838_nugget_2',
                'text': "Ensure 'max_new_tokens' is set to a value that does not exceed the adjusted context length (e.g., 600).",
                'relevant_corpus_ids': [
                    'transformers/src/transformers/generation/utils.py_66208_74709',
                    'langchainjs/docs/core_docs/docs/how_to/query_high_cardinality.ipynb_7337_14397'
                ]
            }
        ]
    })
    
    # Create mock Prediction with real UUIDs from Weaviate
    # This will give partial coverage
    mock_sources = []
    for obj in real_objects.objects:
        mock_source = type('MockSource', (), {'object_id': str(obj.uuid)})()
        mock_sources.append(mock_source)
        print(f"Using real UUID: {obj.uuid} -> dataset_id: {obj.properties.get('dataset_id')}")
    
    mock_prediction = Prediction(
        sources=mock_sources
    )
    
    # Test the metric
    print("\nTesting GEPA Coverage Metric with Real Example")
    print("=" * 70)
    
    # Call the metric (simulating what GEPA would do)
    result = gepa_metric(
        weaviate_client=weaviate_client,
        example=mock_example,
        prediction=mock_prediction,
        trace=None,
        pred_name=None,
        pred_trace=None
    )
    
    print(f"Score: {result.score:.2%}")
    print(f"\nFeedback:\n{result.feedback}")
    print("=" * 70)
    
    weaviate_client.close()

if __name__ == "__main__":
    asyncio.run(main())