import os
from typing import Optional

import weaviate
from weaviate.classes.query import Filter, Metrics
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.outputs.query import QueryReturn

from retrieve_dspy.models import Source, SearchResult

def weaviate_search_tool(
        query: str,
        collection_name: str,
        target_property_name: str,
        retrieved_k: Optional[int] = 5,
        tag_filter_value: Optional[str] = None,
        return_format: str = "string"  # "string", "dict", or "rerank"
):
    """Enhanced search tool that returns results with hybrid scores for reranking."""
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
        )
    )

    collection = weaviate_client.collections.get(collection_name)

    if tag_filter_value:
        search_results = collection.query.hybrid(
            query=query,
            filters=Filter.by_property("tags").contains_any([tag_filter_value]),
            # return_metadata=MetadataQuery(score=True),
            limit=retrieved_k
        )
    else:
        search_results = collection.query.hybrid(
            query=query,
            # return_metadata=MetadataQuery(score=True),
            limit=retrieved_k
        )

    weaviate_client.close()

    # Build source mapping
    object_ids = []
    if search_results.objects:
        for obj in search_results.objects:
            object_ids.append(Source(
                object_id=str(obj.uuid)
            ))

    if return_format == "rerank":
        # Format for RerankResults signature
        search_results_for_rerank = []
        for i, obj in enumerate(search_results.objects):
            # Extract the main content
            content = ""
            if obj.properties and target_property_name in obj.properties:
                content = obj.properties[target_property_name]
            
            # Get the hybrid score from metadata
            # score = obj.metadata.score
            
            search_results_for_rerank.append(SearchResult(
                id=i + 1,  # 1-based indexing
                initial_rank=i + 1,  # Rank based on order returned by Weaviate
                # initial_score=float(score),
                content=content
            ))
        
        return search_results_for_rerank, object_ids
    
    elif return_format == "dict":
        return _dictify_search_results(search_results, view_properties=[target_property_name]), object_ids
    else:
        return _stringify_search_results(search_results, view_properties=[target_property_name]), object_ids

async def async_weaviate_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    retrieved_k: Optional[int] = 10,
    tag_filter_value: Optional[str] = None,
    return_format: str = "string"
):
    """Async version of search tool with hybrid scores."""
    async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=120)  # Values in seconds
        )
    )
    
    await async_client.connect()
    
    try:
        collection = async_client.collections.get(collection_name)
        
        # Request hybrid score metadata
        if tag_filter_value:
            search_results = await collection.query.hybrid(
                query=query,
                filters=Filter.by_property("tags").contains_any([tag_filter_value]),
                # return_metadata=MetadataQuery(score=True),
                limit=retrieved_k
            )
        else:
            search_results = await collection.query.hybrid(
                query=query,
                # return_metadata=MetadataQuery(score=True),
                limit=retrieved_k
            )
        
        object_ids = []
        if search_results.objects:
            for obj in search_results.objects:
                object_ids.append(Source(
                    object_id=str(obj.uuid)
                ))
        
        if return_format == "rerank":
            search_results_for_rerank = []
            for i, obj in enumerate(search_results.objects):
                content = ""
                if obj.properties and target_property_name in obj.properties:
                    content = obj.properties[target_property_name]
                
                # score = obj.metadata.score
                
                search_results_for_rerank.append(SearchResult(
                    id=i + 1,
                    initial_rank=i + 1,
                    # initial_score=float(score),
                    content=content
                ))
            
            return search_results_for_rerank, object_ids
        
        elif return_format == "dict":
            return _dictify_search_results(search_results, view_properties=[target_property_name]), object_ids
        else:
            return _stringify_search_results(search_results, view_properties=[target_property_name]), object_ids
    
    finally:
        await async_client.close()

def _stringify_search_results(search_results: QueryReturn, view_properties=None) -> str:
    """
    Convert Weaviate search results to a readable string format.
    
    Args:
        search_results: The QueryReturn object from Weaviate
        view_properties: List of property names to include (None means include nothing)
                         Can include metadata fields prefixed with underscore
    
    Returns:
        A formatted string representation of the search results
    """
    result_str = f"Found {len(search_results.objects)} results:\n\n"
    
    for i, obj in enumerate(search_results.objects):
        result_str += f"Result {i+1}:\n"
        
        if view_properties:
            if obj.properties:
                properties_to_show = {k: v for k, v in obj.properties.items() if k in view_properties}
                
                if properties_to_show:
                    result_str += "Properties:\n"
                    for key, value in properties_to_show.items():
                        result_str += f"  {key}: {value}\n"
            
            if obj.metadata:
                metadata_fields = []
                for attr in dir(obj.metadata):
                    if attr in view_properties:
                        value = getattr(obj.metadata, attr)
                        if value is not None:
                            metadata_fields.append((attr, value))
                
                if metadata_fields:
                    result_str += "Metadata:\n"
                    for attr, value in metadata_fields:
                        result_str += f"  {attr}: {value}\n"
        
        result_str += "\n"
    
    return result_str

def _dictify_search_results(search_results: QueryReturn, view_properties=None) -> dict[int, str]:
    """
    Convert Weaviate search results to a dictionary with integer keys (1-based).
    
    Args:
        search_results: The QueryReturn object from Weaviate
        view_properties: List of property names to include
    
    Returns:
        A dictionary mapping numeric IDs to formatted search result strings
    """
    result_dict = {}
    
    for i, obj in enumerate(search_results.objects):
        result_id = i + 1  # 1-based indexing
        result_str = f"Result {result_id}:\n"
        
        if view_properties:
            if obj.properties:
                properties_to_show = {k: v for k, v in obj.properties.items() if k in view_properties}
                
                if properties_to_show:
                    result_str += "Properties:\n"
                    for key, value in properties_to_show.items():
                        result_str += f"  {key}: {value}\n"
            
            if obj.metadata:
                metadata_fields = []
                for attr in dir(obj.metadata):
                    if attr in view_properties:
                        value = getattr(obj.metadata, attr)
                        if value is not None:
                            metadata_fields.append((attr, value))
                
                if metadata_fields:
                    result_str += "Metadata:\n"
                    for attr, value in metadata_fields:
                        result_str += f"  {attr}: {value}\n"
        
        result_dict[result_id] = result_str
    
    return result_dict

def get_tag_values(collection_name: str) -> list[str]:
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

    catalog_collection = weaviate_client.collections.get("WeaviateCatalogAgent")

    response = catalog_collection.aggregate.over_all(
        filters=Filter.by_property("reference_collection").equal(collection_name),
        return_metrics=Metrics("tag").text(
            top_occurrences_count=True,
            top_occurrences_value=True,
            min_occurrences=10
        )
    )
    
    # Extract tag values from top occurrences
    tag_values = [
        occurrence.value 
        for occurrence in response.properties["tag"].top_occurrences
    ]

    tags_with_descriptions = {}

    for tag in tag_values:
        response = catalog_collection.query.fetch_objects(
            filters=Filter.by_property("tag").equal(tag),
            limit=1
        )
        for o in response.objects:
            tags_with_descriptions[tag] = o.properties["tag_description"]

    return tags_with_descriptions