import asyncio
import os
from typing import Optional

import weaviate
from weaviate.classes.query import Filter, Metrics, MetadataQuery

from retrieve_dspy.models import ObjectFromDB

def weaviate_search_tool(
        query: str,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,
        return_property_name: Optional[str] = None,
        retrieved_k: Optional[int] = 20,
        return_vector: bool = False,
        return_score: bool = False,
        tag_filter_value: Optional[str] = None,
        search_type: str = "hybrid",
        hybrid_alpha: Optional[float] = None,
) -> list[ObjectFromDB]:
    if weaviate_client is None:
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )
    collection = weaviate_client.collections.get(collection_name)

    '''
    TODO: Add Support for Tag Filtering and Target Vectors with something like `**kwargs`
    if tag_filter_value:
        filter = Filter.by_property("tags").contains_any([tag_filter_value])
    '''
    if search_type == "bm25":
        search_results = collection.query.bm25(
            query=query,
            return_metadata=MetadataQuery(score=return_score),
            include_vector=return_vector,
            limit=retrieved_k
        )
    elif search_type == "vector":
        search_results = collection.query.near_text(
            query=query,
            return_metadata=MetadataQuery(distance=return_score),
            include_vector=return_vector,
            limit=retrieved_k
        )
    else:
        hybrid_kwargs = dict(
            query=query,
            return_metadata=MetadataQuery(score=return_score),
            include_vector=return_vector,
            limit=retrieved_k,
        )
        if hybrid_alpha is not None:
            hybrid_kwargs["alpha"] = hybrid_alpha
        search_results = collection.query.hybrid(**hybrid_kwargs)

    objects: list[ObjectFromDB] = []
    if search_results.objects:
        for rank, obj in enumerate(search_results.objects, start=1):
            object_id = str(obj.properties.get('dataset_id') or obj.uuid)
            content_value = None
            if obj.properties and target_property_name in obj.properties:
                content_value = obj.properties[target_property_name]

            # Extract score: BM25 returns score (higher=better),
            # vector returns distance (lower=better, convert to similarity)
            if return_score:
                if search_type == "vector":
                    score = 1.0 - obj.metadata.distance if obj.metadata.distance is not None else None
                else:
                    score = obj.metadata.score
            else:
                score = None

            objects.append(ObjectFromDB(
                object_id=object_id,
                content=str(content_value) if content_value is not None else "",
                relevance_rank=rank,
                relevance_score=score,
                vector=(obj.vector.get("default") if return_vector and getattr(obj, 'vector', None) else None)
            ))
    return objects

async def async_weaviate_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None,
    return_property_name: Optional[str] = None,
    retrieved_k: Optional[int] = 20,
    return_score: bool = False,
    return_vector: bool = False,
    tag_filter_value: Optional[str] = None,
    search_type: str = "hybrid",
    hybrid_alpha: Optional[float] = None,
) -> list[ObjectFromDB]:
    if weaviate_async_client is None:
        weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        )
        await weaviate_async_client.connect()
    collection = weaviate_async_client.collections.get(collection_name)
    '''
    TODO: Add Support for Tag Filtering and Target Vectors with something like `**kwargs`
    if tag_filter_value:
        filter = Filter.by_property("tags").contains_any([tag_filter_value])
    '''

    if search_type == "bm25":
        search_results = await collection.query.bm25(
            query=query,
            return_metadata=MetadataQuery(score=return_score),
            include_vector=return_vector,
            limit=retrieved_k
        )
    elif search_type == "vector":
        search_results = await collection.query.near_text(
            query=query,
            return_metadata=MetadataQuery(distance=return_score),
            include_vector=return_vector,
            limit=retrieved_k
        )
    else:
        hybrid_kwargs = dict(
            query=query,
            return_metadata=MetadataQuery(score=return_score),
            include_vector=return_vector,
            limit=retrieved_k,
        )
        if hybrid_alpha is not None:
            hybrid_kwargs["alpha"] = hybrid_alpha
        search_results = await collection.query.hybrid(**hybrid_kwargs)

    objects: list[ObjectFromDB] = []
    if search_results.objects:
        for rank, obj in enumerate(search_results.objects, start=1):
            object_id = str(obj.properties.get('dataset_id') or obj.uuid)
            content_value = None
            if obj.properties and target_property_name in obj.properties:
                content_value = obj.properties[target_property_name]

            if return_score:
                if search_type == "vector":
                    score = 1.0 - obj.metadata.distance if obj.metadata.distance is not None else None
                else:
                    score = obj.metadata.score
            else:
                score = None

            objects.append(ObjectFromDB(
                object_id=object_id,
                content=str(content_value) if content_value is not None else "",
                relevance_rank=rank,
                relevance_score=score,
                vector=(obj.vector.get("default") if return_vector and getattr(obj, 'vector', None) else None)
            ))
    return objects

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
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
    )
    sync_results = weaviate_search_tool(
        weaviate_client=weaviate_client,
        query="How do I use Weaviate with Langchain?",
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        return_vector=True,
    )
    print(sync_results)
    print("Testing async search tool...")
    async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    async_results = await async_weaviate_search_tool(
        weaviate_async_client=async_client,
        query="How do I use Weaviate with Langchain?",
        collection_name="FreshstackLangchain",
        target_property_name="docs_text",
        retrieved_k=10,
        return_vector=True,
    )
    print(async_results)

if __name__ == "__main__":
    asyncio.run(main())