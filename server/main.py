"""
FastAPI server for retrieve-dspy retrievers.

Run with: uvicorn server.main:app --reload
Or: python -m server.main
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import weaviate
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from retrieve_dspy import (
    BaseRetriever,
    RAGFusion,
    ConcatenatedQuerySearcher,
    HyDE_QueryExpander,
    PRF_QueryExpander,
    SearchQueryWriter,
)
from retrieve_dspy.retrievers.embeddings_registry import get_embedding_headers
from retrieve_dspy.models import RerankerClient


# ─────────────────────────────────────────────────────────────────────────────
# Request/Response Models
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    k: Optional[int] = Field(None, description="Number of results to return (overrides config)")


class SearchResult(BaseModel):
    object_id: str
    content: str
    relevance_rank: Optional[int] = None
    relevance_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: list[str]
    retriever: str
    total_results: int


class HealthResponse(BaseModel):
    status: str
    retriever: str
    collection: str


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to server-config.yml in the same directory
        config_path = Path(__file__).parent / "server-config.yml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    config: dict = {}
    retriever: Optional[BaseRetriever] = None
    weaviate_async_client: Optional[weaviate.WeaviateAsyncClient] = None


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Reranker Client Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_dspy_reranker_module(dspy_config: dict):
    """Create a DSPy reranker module based on configuration.
    
    Args:
        dspy_config: Dictionary with:
            - signature: "binary" (AssessRelevance) or "numeric" (ScoreRelevance)
            - verbose: Whether to use verbose signature
    """
    import dspy
    from retrieve_dspy.signatures import (
        AssessRelevance, 
        ScoreRelevance, 
        VerboseScoreRelevance
    )
    
    signature_type = dspy_config.get("signature", "numeric")
    verbose = dspy_config.get("verbose", True)
    
    if signature_type == "numeric":
        signature = VerboseScoreRelevance if verbose else ScoreRelevance
    else:  # binary
        signature = AssessRelevance
    
    return dspy.Predict(signature)


def create_reranker_clients(reranker_config: dict) -> list[RerankerClient]:
    """Create reranker clients based on configuration."""
    clients = []
    providers = reranker_config.get("providers", [])
    
    for provider in providers:
        if provider == "cohere":
            import cohere
            api_key = os.getenv("COHERE_API_KEY")
            if api_key:
                clients.append(RerankerClient(name="cohere", client=cohere.Client(api_key)))
        elif provider == "voyage":
            import voyageai
            api_key = os.getenv("VOYAGE_API_KEY")
            if api_key:
                clients.append(RerankerClient(name="voyage", client=voyageai.Client(api_key=api_key)))
        elif provider == "dspy":
            # Create DSPy LLM-based reranker module
            dspy_config = reranker_config.get("dspy", {})
            module = create_dspy_reranker_module(dspy_config)
            clients.append(RerankerClient(name="dspy", client=module))
            print(f"   Created DSPy reranker (signature: {dspy_config.get('signature', 'numeric')})")
    
    return clients


# ─────────────────────────────────────────────────────────────────────────────
# Retriever Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_retriever(config: dict) -> BaseRetriever:
    """Create a retriever instance based on configuration."""
    retriever_config = config["retriever"]
    weaviate_config = config["weaviate"]
    
    # Get active retriever name and its params
    retriever_name = retriever_config.get("active") or retriever_config.get("name")
    retriever_params = retriever_config.get(retriever_name, {}) or retriever_config.get("params", {})
    search_type = weaviate_config.get("search_type", "hybrid")


    print(f"📋 Config loaded: retriever={retriever_name}")
    print(f"   retriever_params={retriever_params}")

    if retriever_name == "BaseRetriever":
        return BaseRetriever(
            collection_name=weaviate_config["collection_name"],
            target_property_name=weaviate_config.get("target_property_name", "content"),
            retrieved_k=retriever_params.get("retrieved_k", 20),
            verbose=retriever_params.get("verbose", False),
            search_only=retriever_params.get("search_only", True),
            diversity_weight=retriever_params.get("diversity_weight", 0),
            search_type=search_type,
        )
    elif retriever_name == "RAGFusion":
        # Get fusion strategy (with backwards compatibility for use_rrf)
        fusion_strategy = retriever_params.get("fusion_strategy", "rrf")
        if fusion_strategy == "rrf" and not retriever_params.get("use_rrf", True):
            fusion_strategy = "interleave"
        
        # Build cross-encoder reranker clients if configured
        reranker_config = retriever_params.get("cross_encoder", {})
        reranker_clients = None
        
        # CE-Gated requires cross-encoder clients
        ce_requires_reranker = fusion_strategy == "ce_gated"
        ce_enabled = reranker_config.get("enabled", False)
        
        if ce_enabled or ce_requires_reranker:
            reranker_clients = create_reranker_clients(reranker_config)
            
            if not reranker_clients:
                providers = reranker_config.get("providers", [])
                missing_keys = []
                if "cohere" in providers and not os.getenv("COHERE_API_KEY"):
                    missing_keys.append("COHERE_API_KEY")
                if "voyage" in providers and not os.getenv("VOYAGE_API_KEY"):
                    missing_keys.append("VOYAGE_API_KEY")
                
                if ce_requires_reranker:
                    raise ValueError(
                        f"CE-Gated pooling requires reranker clients but none could be created. "
                        f"Missing environment variables: {missing_keys or 'unknown'}. "
                        f"Configured providers: {providers}"
                    )
                else:
                    print(f"⚠️  Warning: Cross-encoder enabled but no clients created. Missing: {missing_keys}")
        
        # CE-Gated pooling config
        ce_gated_config = retriever_params.get("ce_gated", {})
        
        print(f"   fusion_strategy={fusion_strategy}")
        if fusion_strategy == "ce_gated":
            print(f"   ce_gated_alpha={ce_gated_config.get('alpha', 0.7)}")
            print(f"   ce_gated_aggregation={ce_gated_config.get('aggregation', 'max')}")
            print(f"   reranker_clients={[c.name for c in reranker_clients] if reranker_clients else None}")
        
        return RAGFusion(
            collection_name=weaviate_config["collection_name"],
            target_property_name=weaviate_config.get("target_property_name", "content"),
            number_of_queries=retriever_params.get("number_of_queries", 5),
            retrieved_k=retriever_params.get("retrieved_k", 20),
            reranked_k=retriever_params.get("reranked_k", 200),
            rrf_k=retriever_params.get("rrf_k", 60),
            fusion_strategy=fusion_strategy,
            # Cross-encoder config (for post-fusion reranking with rrf/interleave)
            use_cross_encoder=reranker_config.get("enabled", False) and fusion_strategy != "ce_gated",
            reranker_clients=reranker_clients,
            reranker_provider=reranker_config.get("provider"),
            ce_rerank_k=reranker_config.get("top_k"),
            # CE-Gated pooling config
            ce_gated_alpha=ce_gated_config.get("alpha", 0.7),
            ce_gated_aggregation=ce_gated_config.get("aggregation", "max"),
            verbose=retriever_params.get("verbose", False),
            verbose_signature=retriever_params.get("verbose_signature", True),
        )
    elif retriever_name == "ConcatenatedQuerySearcher":
        return ConcatenatedQuerySearcher(
            collection_name=weaviate_config["collection_name"],
            target_property_name=weaviate_config.get("target_property_name", "content"),
            retrieved_k=retriever_params.get("retrieved_k", 20),
            number_of_queries=retriever_params.get("number_of_queries", 5),
            final_k=retriever_params.get("final_k"),
            verbose=retriever_params.get("verbose", False),
            verbose_signature=retriever_params.get("verbose_signature", True),
            search_only=retriever_params.get("search_only", True),
            diversity_weight=retriever_params.get("diversity_weight", 0.0),
            diversity_strategy=retriever_params.get("diversity_strategy", "mmr"),
        )
    elif retriever_name == "HyDE_QueryExpander":
        return HyDE_QueryExpander(
            collection_name=weaviate_config["collection_name"],
            target_property_name=weaviate_config.get("target_property_name", "content"),
            retrieved_k=retriever_params.get("retrieved_k", 20),
            verbose=retriever_params.get("verbose", False),
            search_only=retriever_params.get("search_only", True),
            search_type=search_type,
        )
    elif retriever_name == "PRF_QueryExpander":
        return PRF_QueryExpander(
            collection_name=weaviate_config["collection_name"],
            target_property_name=weaviate_config.get("target_property_name", "content"),
            retrieved_k=retriever_params.get("retrieved_k", 20),
            verbose=retriever_params.get("verbose", False),
            search_only=retriever_params.get("search_only", True),
        )
    elif retriever_name == "SearchQueryWriter":
        return SearchQueryWriter(
            collection_name=weaviate_config["collection_name"],
            target_property_name=weaviate_config.get("target_property_name", "content"),
            retrieved_k=retriever_params.get("retrieved_k", 20),
            verbose=retriever_params.get("verbose", False),
            search_only=retriever_params.get("search_only", True),
        )
    else:
        raise ValueError(f"Unsupported retriever: {retriever_name}. Supported: 'BaseRetriever', 'RAGFusion', 'ConcatenatedQuerySearcher', 'HyDE_QueryExpander', 'PRF_QueryExpander', 'SearchQueryWriter'.")


def get_weaviate_async_client(config: dict) -> weaviate.WeaviateAsyncClient:
    """Create and return a Weaviate async client (must call connect() before use)."""
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url or not weaviate_api_key:
        raise ValueError(
            "WEAVIATE_URL and WEAVIATE_API_KEY environment variables must be set"
        )

    embedding_model = config.get("weaviate", {}).get(
        "embedding_model", "weaviate/Snowflake/snowflake-arctic-embed-l-v2.0"
    )
    headers = get_embedding_headers(embedding_model)

    return weaviate.use_async_with_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
        headers=headers,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - setup and teardown."""
    # Startup
    config_path = os.getenv("RETRIEVER_CONFIG_PATH")
    state.config = load_config(config_path)
    state.retriever = create_retriever(state.config)
    state.weaviate_async_client = get_weaviate_async_client(state.config)
    await state.weaviate_async_client.connect()
    
    retriever_name = state.config['retriever'].get('active') or state.config['retriever'].get('name')
    print(f"🚀 Server starting with retriever: {retriever_name}")
    print(f"📦 Collection: {state.config['weaviate']['collection_name']}")
    
    yield
    
    # Shutdown
    if state.weaviate_async_client:
        await state.weaviate_async_client.close()
    print("👋 Server shutting down")


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """Create FastAPI application with optional config path."""
    if config_path:
        os.environ["RETRIEVER_CONFIG_PATH"] = config_path
    
    return FastAPI(
        title="retrieve-dspy Server",
        description="FastAPI server for retrieve-dspy retrievers",
        version="0.1.0",
        lifespan=lifespan,
    )


# Create default app instance
app = create_app()


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        retriever=state.config.get("retriever", {}).get("active") or state.config.get("retriever", {}).get("name", "unknown"),
        collection=state.config.get("weaviate", {}).get("collection_name", "unknown"),
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Execute a search query using the configured retriever.
    
    Returns a list of search results ranked by relevance.
    """
    if state.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    if state.weaviate_async_client is None:
        raise HTTPException(status_code=503, detail="Weaviate client not initialized")
    
    # Use aforward for non-blocking async execution
    response = await state.retriever.aforward(
        question=request.query,
        weaviate_async_client=state.weaviate_async_client,
    )
        
    # Convert sources to response format
    results = [source.object_id for source in response.sources]

    print(f"Sending {len(results)} results to the client")
        
    return SearchResponse(
        query=request.query,
        results=results,
        retriever=state.config["retriever"].get("active") or state.config["retriever"].get("name"),
        total_results=len(results),
    )


@app.get("/config")
async def get_config():
    """Return the current server configuration."""
    return {
        "retriever": state.config.get("retriever", {}),
        "weaviate": {
            "collection_name": state.config.get("weaviate", {}).get("collection_name"),
            "target_property_name": state.config.get("weaviate", {}).get("target_property_name"),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    
    config = load_config()
    server_config = config.get("server", {})
    
    uvicorn.run(
        "server.main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=True,
    )

