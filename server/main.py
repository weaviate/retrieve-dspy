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

from retrieve_dspy import HybridSearch


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
    retriever: Optional[HybridSearch] = None
    weaviate_client: Optional[weaviate.WeaviateClient] = None


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Retriever Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_retriever(config: dict) -> HybridSearch:
    """Create a retriever instance based on configuration."""
    retriever_name = config["retriever"]["name"]
    retriever_params = config["retriever"].get("params", {})
    weaviate_config = config["weaviate"]
    
    if retriever_name == "HybridSearch":
        return HybridSearch(
            collection_name=weaviate_config["collection_name"],
            target_property_name=weaviate_config.get("target_property_name", "content"),
            retrieved_k=retriever_params.get("retrieved_k", 20),
            verbose=retriever_params.get("verbose", False),
            search_only=retriever_params.get("search_only", True),
        )
    else:
        raise ValueError(f"Unsupported retriever: {retriever_name}. Currently only 'HybridSearch' is supported.")


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Create and return a Weaviate client."""
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        raise ValueError(
            "WEAVIATE_URL and WEAVIATE_API_KEY environment variables must be set"
        )
    
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
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
    state.weaviate_client = get_weaviate_client()
    
    print(f"🚀 Server starting with retriever: {state.config['retriever']['name']}")
    print(f"📦 Collection: {state.config['weaviate']['collection_name']}")
    
    yield
    
    # Shutdown
    if state.weaviate_client:
        state.weaviate_client.close()
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
        retriever=state.config.get("retriever", {}).get("name", "unknown"),
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
    
    if state.weaviate_client is None:
        raise HTTPException(status_code=503, detail="Weaviate client not initialized")
    
    try:
        # Override k if provided in request
        if request.k is not None:
            original_k = state.retriever.retrieved_k
            state.retriever.retrieved_k = request.k
        
        # Execute search
        response = state.retriever.forward(
            question=request.query,
            weaviate_client=state.weaviate_client,
        )
        
        # Restore original k
        if request.k is not None:
            state.retriever.retrieved_k = original_k
        
        # Convert sources to response format
        results= [source.object_id for source in response.sources]
        
        return SearchResponse(
            query=request.query,
            results=results,
            retriever=state.config["retriever"]["name"],
            total_results=len(results),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

