"""
Test script for the retrieve-dspy server.

Usage:
    1. Start the server: uvicorn server.main:app --reload
    2. Run this script: python scripts/test-server.py
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print()
    return response.status_code == 200


def test_config():
    """Test the config endpoint."""
    print("Testing /config endpoint...")
    response = requests.get(f"{BASE_URL}/config")
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    print()
    return response.status_code == 200


def test_search(query: str, k: int = 5):
    """Test the search endpoint."""
    print(f"Testing /search endpoint with query: '{query}'...")
    response = requests.post(
        f"{BASE_URL}/search",
        json={"query": query, "k": k}
    )
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  Retriever: {data['retriever']}")
        print(f"  Total results: {data['total_results']}")
        print(f"  Results:")
        for i, result in enumerate(data["results"][:3], 1):
            content_preview = result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
            print(f"    {i}. [Score: {result['relevance_score']:.4f}] {content_preview}")
    else:
        print(f"  Error: {response.text}")
    
    print()
    return response.status_code == 200


def main():
    print("=" * 60)
    print("retrieve-dspy Server Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Test health
    results.append(("Health", test_health()))
    
    # Test config
    results.append(("Config", test_config()))
    
    # Test search
    results.append(("Search", test_search("What is retrieval augmented generation?", k=5)))
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()

