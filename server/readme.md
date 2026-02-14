# retrieve-dspy Server

FastAPI server for retrieve-dspy retrievers.

## Prerequisites

Set the following environment variables:

- `WEAVIATE_URL` - Your Weaviate cluster URL
- `WEAVIATE_API_KEY` - Your Weaviate API key
- `VOYAGE_API_KEY` - (Optional) Required if using Voyage reranker
- `COHERE_API_KEY` - (Optional) Required if using Cohere reranker

## Starting the Server

### Using `uv` (Recommended)

From root,

```bash
uv run uvicorn server.main:app --reload
```

Or run the module directly:

```bash
uv run python -m server.main
```

### Using `python` directly

```bash
uvicorn server.main:app --reload
```

Or:

```bash
python -m server.main
```

## Configuration

The server configuration is loaded from `server-config.yml` in the server directory. You can override the config path by setting the `RETRIEVER_CONFIG_PATH` environment variable.

Default server settings:
- Host: `0.0.0.0`
- Port: `8000`

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /search` - Execute a search query
- `GET /config` - Get current server configuration

## Example Usage

Once the server is running, you can test it:

```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query here"}'
```
