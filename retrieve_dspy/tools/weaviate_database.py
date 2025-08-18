import asyncio
import os
from typing import Literal, Optional

import weaviate
from weaviate.classes.query import Filter, Metrics, MetadataQuery
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.outputs.query import QueryReturn

from retrieve_dspy.models import Source, SourceWithContentAndVector, SearchResult

RETURN_FORMATS = ["string", "dict", "rerank", "vectors"]

# Extend to add `return_properties`
def weaviate_search_tool(
        query: str,
        collection_name: str,
        target_property_name: str,
        return_property_name: Optional[str] = None,
        retrieved_k: Optional[int] = 5,
        return_score: bool = False,
        return_vector: bool = False,
        tag_filter_value: Optional[str] = None,
        return_format: Literal["string", "dict", "rerank", "vectors"] = "string"
):
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    )

    collection = weaviate_client.collections.get(collection_name)
    '''
    return_metadata = None
    if return_score:
        return_metadata = MetadataQuery(score=return_score)
    '''
    
    if return_property_name is None:
        return_property_name = target_property_name

    '''
    if tag_filter_value:
        filter = Filter.by_property("tags").contains_any([tag_filter_value])
    '''

    '''
    search_results = collection.query.hybrid(
        query=query,
        limit=retrieved_k,
        return_metadata=return_metadata,
        return_properties=return_properties,
        include_vector=return_vector
    )
    '''
    print(f"SEARCHING WITH K = {retrieved_k}")
    search_results = collection.query.hybrid(
        query=query,
        limit=retrieved_k,
        target_vector=target_property_name
    )

    weaviate_client.close()

    # Build `Source` list of object IDs
    object_ids: list[Source] = []
    if search_results.objects:
        for obj in search_results.objects:
            # Instead of UUID, use dataset_id directly
            dataset_id = obj.properties.get('dataset_id')
            if dataset_id:
                object_ids.append(Source(object_id=str(dataset_id)))

    if return_format == "vectors":
        sources_with_content_and_vector: list[SourceWithContentAndVector] = []
        for obj in search_results.objects:
            sources_with_content_and_vector.append(SourceWithContentAndVector(
                object_id=str(obj.uuid),
                content=obj.properties[target_property_name],
                vector=obj.vector["default"] # update with named vectors
            ))
        return sources_with_content_and_vector, object_ids

    if return_format == "rerank":
        search_results_for_rerank: list[SearchResult] = []
        for i, obj in enumerate(search_results.objects):
            content = ""
            if obj.properties and return_property_name in obj.properties:
                content = obj.properties[return_property_name]
            
            search_results_for_rerank.append(SearchResult(
                id=i + 1,
                initial_rank=i + 1,
                # initial_score=float(score), # TODO: this was added to test InsertRank Seetharam et al. 2025
                content=content
            ))
        
        return search_results_for_rerank, object_ids
    
    elif return_format == "dict":
        return _dictify_search_results(search_results, view_properties=[return_property_name]), object_ids
    else:
        return _stringify_search_results(search_results, view_properties=[return_property_name]), object_ids

async def async_weaviate_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    return_property_name: Optional[str] = None,
    retrieved_k: Optional[int] = 10,
    return_score: bool = False,
    return_vector: bool = False,
    tag_filter_value: Optional[str] = None,
    return_format: Literal["string", "dict", "rerank", "vectors"] = "string"
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

        return_metadata = None
        if return_score:
            return_metadata = MetadataQuery(score=return_score)
        
        if return_property_name is None:
            return_property_name = target_property_name
        return_properties = [return_property_name]

        '''
        if tag_filter_value:
            filter = Filter.by_property("tags").contains_any([tag_filter_value])
        '''
        kwargs = dict(
            query=query,
            limit=retrieved_k,
            return_metadata=return_metadata,
            return_properties=return_properties,
            include_vector=return_vector,
            target_vector=target_property_name
        )
        
        search_results = await collection.query.hybrid(**kwargs)
        
        object_ids = []
        if search_results.objects:
            for obj in search_results.objects:
                object_ids.append(Source(
                    object_id=str(obj.uuid)
                ))
        
        if return_format == "vectors":
            sources_with_content_and_vector: list[SourceWithContentAndVector] = []
            for obj in search_results.objects:
                sources_with_content_and_vector.append(SourceWithContentAndVector(
                    object_id=str(obj.uuid),
                    content=obj.properties[target_property_name],
                    vector=obj.vector["default"] # update with named vectors
                ))
            return sources_with_content_and_vector, object_ids
        
        if return_format == "rerank":
            search_results_for_rerank = []
            for i, obj in enumerate(search_results.objects):
                content = ""
                if obj.properties and return_property_name in obj.properties:
                    content = obj.properties[return_property_name]
                
                # score = obj.metadata.score
                
                search_results_for_rerank.append(SearchResult(
                    id=i + 1,
                    initial_rank=i + 1,
                    # initial_score=float(score),
                    content=content
                ))
            
            return search_results_for_rerank, object_ids
        
        elif return_format == "dict":
            return _dictify_search_results(search_results, view_properties=[return_property_name]), object_ids
        else:
            return _stringify_search_results(search_results, view_properties=[return_property_name]), object_ids
    
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

async def main():
    print("Testing sync search tool...")
    sync_results = weaviate_search_tool(
        query="How do I use Weaviate with Langchain?",
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        return_score=True,
        return_vector=True,
        return_format="vectors"
    )
    print(sync_results)
    print("Testing async search tool...")
    async_results = await async_weaviate_search_tool(
        query="How do I use Weaviate with Langchain?",
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        return_score=True,
        return_vector=True,
        return_format="vectors"
    )
    print(async_results)

if __name__ == "__main__":
    asyncio.run(main())