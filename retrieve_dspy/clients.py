import os
import weaviate
import cohere
import voyageai

def get_weaviate_client():
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )

async def get_weaviate_async_client():
    weaviate_async_client = weaviate.use_async_with_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    )
    await weaviate_async_client.connect()
    return weaviate_async_client

def get_cohere_client():
    return cohere.ClientV2(os.getenv("COHERE_API_KEY"))

def get_voyage_client():
    return voyageai.Client(os.getenv("VOYAGE_API_KEY"))