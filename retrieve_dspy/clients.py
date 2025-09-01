import os
import weaviate
import cohere
import voyageai

from retrieve_dspy.models import RerankerClient

def get_weaviate_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

async def get_weaviate_async_client() -> weaviate.WeaviateAsyncClient:
    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    await weaviate_async_client.connect()
    return weaviate_async_client

def get_cohere_client() -> RerankerClient:
    return RerankerClient(name="cohere", client=cohere.ClientV2(os.getenv("COHERE_API_KEY")))

def get_voyage_client() -> RerankerClient:
    return RerankerClient(name="voyage", client=voyageai.Client(os.getenv("VOYAGE_API_KEY")))