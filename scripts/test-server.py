"""
Test script for the retrieve-dspy server.

Usage:
    1. Start the server: uvicorn server.main:app --reload
    2. Run this script: python scripts/test-server.py

Tests both basic functionality and async concurrency.
"""

import asyncio
import time
from typing import Optional

import httpx

BASE_URL = "http://localhost:8000"


async def test_health(client: httpx.AsyncClient) -> bool:
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = await client.get(f"{BASE_URL}/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print()
    return response.status_code == 200


async def test_config(client: httpx.AsyncClient) -> bool:
    """Test the config endpoint."""
    print("Testing /config endpoint...")
    response = await client.get(f"{BASE_URL}/config")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print()
    return response.status_code == 200


async def test_search(client: httpx.AsyncClient, query: str, k: int = 5) -> bool:
    """Test the search endpoint."""
    print(f"Testing /search endpoint with query: '{query}'...")
    response = await client.post(
        f"{BASE_URL}/search",
        json={"query": query, "k": k}
    )
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Retriever: {data['retriever']}")
        print(f"  Total results: {data['total_results']}")
        print("  First 25 characters of each result ID:")
        for idx, result in enumerate(data.get('results', [])[:5]):
            first25 = result[:25] if isinstance(result, str) else ""
            print(f"    Result {idx+1}: {repr(first25)}")
    else:
        print(f"  Error: {response.text}")
    
    print()
    return response.status_code == 200


async def single_search_timed(client: httpx.AsyncClient, query: str, request_id: int) -> tuple[int, float, bool]:
    """Execute a single search and return (request_id, duration, success)."""
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{BASE_URL}/search",
            json={"query": query, "k": 5}
        )
        duration = time.perf_counter() - start
        success = response.status_code == 200
        return (request_id, duration, success)
    except Exception as e:
        duration = time.perf_counter() - start
        print(f"  Request {request_id} failed: {e}")
        return (request_id, duration, False)


async def test_async_concurrency(client: httpx.AsyncClient, num_requests: int = 5) -> bool:
    """
    Test that async requests are handled concurrently.
    
    If the server is truly async, N concurrent requests should complete in 
    roughly the same time as 1 request (plus overhead), not N times longer.
    """
    print(f"Testing async concurrency with {num_requests} concurrent requests...")
    print()
    
    queries = [
        "What is retrieval augmented generation?",
        "How do vector databases work?",
        "Explain semantic search techniques",
        "What are embedding models?",
        "How does reranking improve search results?",
    ]
    
    # First, measure a single request to get baseline
    print("  Step 1: Measuring single request baseline...")
    single_start = time.perf_counter()
    _, single_duration, single_success = await single_search_timed(client, queries[0], 0)
    print(f"    Single request took: {single_duration:.2f}s")
    print()
    
    if not single_success:
        print("  ⚠ Single request failed, skipping concurrency test")
        return False
    
    # Now send multiple requests concurrently
    print(f"  Step 2: Sending {num_requests} requests concurrently...")
    concurrent_start = time.perf_counter()
    
    tasks = [
        single_search_timed(client, queries[i % len(queries)], i + 1)
        for i in range(num_requests)
    ]
    results = await asyncio.gather(*tasks)
    
    concurrent_duration = time.perf_counter() - concurrent_start
    
    # Report individual request timings
    print(f"    Individual request durations:")
    all_success = True
    for req_id, duration, success in sorted(results):
        status = "✓" if success else "✗"
        print(f"      Request {req_id}: {duration:.2f}s {status}")
        if not success:
            all_success = False
    
    print()
    print(f"    Total wall-clock time for {num_requests} concurrent requests: {concurrent_duration:.2f}s")
    print(f"    Expected if sequential (blocking): ~{single_duration * num_requests:.2f}s")
    print(f"    Expected if concurrent (async): ~{single_duration:.2f}s + overhead")
    print()
    
    # Calculate speedup
    sequential_estimate = single_duration * num_requests
    speedup = sequential_estimate / concurrent_duration if concurrent_duration > 0 else 0
    
    # If speedup is > 1.5x, async is working (some overhead is expected)
    async_working = speedup > 1.5 and all_success
    
    if async_working:
        print(f"  \033[92m✓ Async is working! Speedup: {speedup:.1f}x\033[0m")
    elif not all_success:
        print(f"  \033[91m✗ Some requests failed\033[0m")
    else:
        print(f"  \033[93m⚠ Speedup only {speedup:.1f}x - async may not be working correctly\033[0m")
    
    print()
    return async_working


async def main():
    print("=" * 60)
    print("retrieve-dspy Server Test")
    print("=" * 60)
    print()
    
    # Use a single client with connection pooling for all requests
    async with httpx.AsyncClient(timeout=120.0) as client:
        results = []
        
        # Test health
        results.append(("Health", await test_health(client)))
        
        # Test config
        results.append(("Config", await test_config(client)))
        
        # Test single search
        results.append(("Search", await test_search(client, "What is retrieval augmented generation?", k=5)))
        
        # Test async concurrency
        results.append(("Async Concurrency", await test_async_concurrency(client, num_requests=5)))
        
        # Summary
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        for name, passed in results:
            if passed:
                status = "\033[92m✓ PASS\033[0m"
            else:
                status = "\033[91m✗ FAIL\033[0m"
            print(f"  {name}: {status}")


if __name__ == "__main__":
    asyncio.run(main())

