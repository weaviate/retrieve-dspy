import os

PROVIDER_HEADERS: dict[str, tuple[str, str]] = {
    "voyageai": ("X-VoyageAI-API-Key", "VOYAGE_API_KEY"),
    "openai": ("X-OpenAI-Api-Key", "OPENAI_API_KEY"),
    "cohere": ("X-Cohere-Api-Key", "COHERE_API_KEY"),
}


def parse_embedding_provider(embedding_model: str) -> str:
    return embedding_model.split("/")[0]


def get_embedding_headers(embedding_model: str) -> dict[str, str]:
    provider = parse_embedding_provider(embedding_model)
    if provider not in PROVIDER_HEADERS:
        return {}
    header_name, env_var = PROVIDER_HEADERS[provider]
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"Embedding model '{embedding_model}' requires "
            f"{env_var} environment variable to be set"
        )
    return {header_name: api_key}
