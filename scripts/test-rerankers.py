"""
Test script for comparing different reranker providers.

Usage:
    python scripts/test-rerankers.py

Tests Cohere, Voyage, and DSPy LLM-based rerankers using toy data.
Set environment variables for the providers you want to test:
    - COHERE_API_KEY: For Cohere reranker
    - VOYAGE_API_KEY: For Voyage reranker
    - OPENAI_API_KEY (or other LLM): For DSPy LLM reranker

The script will skip providers that don't have API keys configured.
"""

import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy

from retrieve_dspy.models import RerankItem
from retrieve_dspy.signatures import VerboseScoreRelevance, AssessRelevance
from retrieve_dspy.retrievers.common.call_ce_ranker import (
    make_cohere_reranker,
    make_voyage_reranker,
    make_dspy_reranker,
)


# ─────────────────────────────────────────────────────────────────────────────
# Toy Test Data
# ─────────────────────────────────────────────────────────────────────────────

TOY_QUERIES = [
    "What is retrieval augmented generation and how does it work?",
    "How do I fix a Python ImportError?",
    "Best practices for React component design",
]

TOY_DOCUMENTS = [
    # Doc 0: Highly relevant to RAG query
    "Retrieval Augmented Generation (RAG) is a technique that combines large language models with external knowledge retrieval. It works by first retrieving relevant documents from a knowledge base using semantic search, then feeding those documents as context to an LLM to generate accurate, grounded responses. RAG helps reduce hallucinations and keeps responses up-to-date.",
    
    # Doc 1: Somewhat relevant to RAG query (mentions LLMs but not RAG specifically)
    "Large language models have revolutionized natural language processing. They can generate text, answer questions, and perform various NLP tasks. However, they sometimes produce incorrect information, known as hallucinations, when they don't have access to current data.",
    
    # Doc 2: Highly relevant to Python ImportError query
    "To fix a Python ImportError, first check that the module is installed using 'pip list'. If not installed, run 'pip install <module_name>'. Also verify you're using the correct Python environment. Common causes include typos in import statements, circular imports, and missing __init__.py files in packages.",
    
    # Doc 3: Tangentially relevant to Python query
    "Python is a versatile programming language used for web development, data science, and automation. It has a rich ecosystem of libraries available through pip, the Python package manager.",
    
    # Doc 4: Highly relevant to React query
    "React component best practices include: keeping components small and focused, using functional components with hooks, lifting state up when needed, using proper prop types or TypeScript, memoizing expensive computations, and following the single responsibility principle. Avoid deeply nested component hierarchies.",
    
    # Doc 5: Somewhat relevant to React query
    "React is a JavaScript library for building user interfaces. It uses a virtual DOM for efficient updates and supports component-based architecture. JSX allows you to write HTML-like syntax in JavaScript.",
    
    # Doc 6: Irrelevant to all queries
    "The weather forecast for tomorrow shows sunny skies with temperatures reaching 75°F. There's a 10% chance of rain in the evening. Perfect conditions for outdoor activities.",
    
    # Doc 7: Irrelevant to all queries
    "The best pizza in New York can be found at several locations. Traditional thin-crust pizza remains popular, but newer styles like Detroit-style and Neapolitan have gained followers.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Reranker Setup Functions
# ─────────────────────────────────────────────────────────────────────────────

def setup_cohere_reranker() -> Optional[callable]:
    """Set up Cohere reranker if API key is available."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    
    try:
        import cohere
        client = cohere.Client(api_key)
        return make_cohere_reranker(client, model="rerank-v3.5")
    except ImportError:
        print("  ⚠ cohere package not installed")
        return None
    except Exception as e:
        print(f"  ⚠ Failed to initialize Cohere: {e}")
        return None


def setup_voyage_reranker() -> Optional[callable]:
    """Set up Voyage reranker if API key is available."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return None
    
    try:
        import voyageai
        client = voyageai.Client(api_key=api_key)
        return make_voyage_reranker(client, model="rerank-2")
    except ImportError:
        print("  ⚠ voyageai package not installed")
        return None
    except Exception as e:
        print(f"  ⚠ Failed to initialize Voyage: {e}")
        return None


def setup_dspy_reranker(score_type: str = "numeric") -> Optional[callable]:
    """Set up DSPy LLM reranker."""
    # Check if DSPy has a configured LM
    try:
        # Try to get the default LM
        lm = dspy.settings.lm
        if lm is None:
            # Try to configure a default LM
            openai_key = os.getenv("OPENAI_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            
            if openai_key:
                lm = dspy.LM("openai/gpt-4o-mini")
                dspy.configure(lm=lm)
            elif anthropic_key:
                lm = dspy.LM("anthropic/claude-3-haiku-20240307")
                dspy.configure(lm=lm)
            else:
                print("  ⚠ No LLM API key found (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
                return None
        
        # Create the DSPy module with appropriate signature
        if score_type == "numeric":
            module = dspy.Predict(VerboseScoreRelevance)
        else:
            module = dspy.Predict(AssessRelevance)
        
        return make_dspy_reranker(module, score_type=score_type)
        
    except Exception as e:
        print(f"  ⚠ Failed to initialize DSPy reranker: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Test Functions
# ─────────────────────────────────────────────────────────────────────────────

def run_reranker_test(
    reranker: callable, 
    query: str, 
    documents: list[str], 
    provider_name: str
) -> list[RerankItem]:
    """Run a single reranker test and return results."""
    try:
        results = reranker(query, documents, top_k=len(documents))
        return results
    except Exception as e:
        print(f"  ⚠ {provider_name} failed: {e}")
        return []


def format_ranking(results: list[RerankItem], documents: list[str], max_preview: int = 50) -> str:
    """Format ranking results for display."""
    lines = []
    for rank, item in enumerate(results, 1):
        doc_preview = documents[item.index][:max_preview].replace('\n', ' ')
        if len(documents[item.index]) > max_preview:
            doc_preview += "..."
        lines.append(f"    {rank}. [Doc {item.index}] Score: {item.relevance_score:.4f} | {doc_preview}")
    return "\n".join(lines)


def compare_rankings(
    rankings: dict[str, list[RerankItem]], 
    documents: list[str]
) -> None:
    """Compare rankings across different providers."""
    if len(rankings) < 2:
        return
    
    print("\n  📊 Ranking Comparison:")
    print("  " + "-" * 70)
    
    # Get top-3 from each provider
    for provider, results in rankings.items():
        top_3_indices = [r.index for r in results[:3]]
        print(f"    {provider:12} Top-3: {top_3_indices}")
    
    # Calculate agreement metrics
    providers = list(rankings.keys())
    for i, p1 in enumerate(providers):
        for p2 in providers[i+1:]:
            r1 = [r.index for r in rankings[p1]]
            r2 = [r.index for r in rankings[p2]]
            
            # Top-1 agreement
            top1_agree = r1[0] == r2[0] if r1 and r2 else False
            
            # Top-3 overlap
            top3_overlap = len(set(r1[:3]) & set(r2[:3]))
            
            # Kendall-tau-like metric (simplified: count inversions in top-5)
            inversions = 0
            for idx in range(min(5, len(r1))):
                if idx < len(r2):
                    pos_in_r2 = r2.index(r1[idx]) if r1[idx] in r2 else len(r2)
                    inversions += abs(idx - pos_in_r2)
            
            print(f"    {p1} vs {p2}: Top-1 {'✓' if top1_agree else '✗'}, Top-3 overlap: {top3_overlap}/3, Inversions: {inversions}")


def test_query(
    query: str, 
    documents: list[str], 
    rerankers: dict[str, callable],
    query_idx: int
) -> None:
    """Test all rerankers on a single query."""
    print(f"\n{'='*70}")
    print(f"Query {query_idx + 1}: {query[:60]}...")
    print("=" * 70)
    
    rankings = {}
    
    for provider_name, reranker in rerankers.items():
        print(f"\n  🔍 {provider_name}:")
        results = run_reranker_test(reranker, query, documents, provider_name)
        
        if results:
            rankings[provider_name] = results
            print(format_ranking(results, documents))
    
    compare_rankings(rankings, documents)


def main():
    print("=" * 70)
    print("Reranker Comparison Test")
    print("=" * 70)
    print()
    print("This script compares rankings from different reranker providers")
    print("using toy data to verify they're working correctly.")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Setup rerankers
    # ─────────────────────────────────────────────────────────────────────────
    print("Setting up rerankers...")
    print()
    
    rerankers = {}
    
    # Cohere
    print("  Cohere: ", end="")
    cohere_reranker = setup_cohere_reranker()
    if cohere_reranker:
        rerankers["Cohere"] = cohere_reranker
        print("✓ Ready")
    else:
        print("✗ Skipped (no COHERE_API_KEY)")
    
    # Voyage
    print("  Voyage: ", end="")
    voyage_reranker = setup_voyage_reranker()
    if voyage_reranker:
        rerankers["Voyage"] = voyage_reranker
        print("✓ Ready")
    else:
        print("✗ Skipped (no VOYAGE_API_KEY)")
    
    # DSPy Numeric
    print("  DSPy (numeric): ", end="")
    dspy_numeric = setup_dspy_reranker(score_type="numeric")
    if dspy_numeric:
        rerankers["DSPy-Numeric"] = dspy_numeric
        print("✓ Ready")
    else:
        print("✗ Skipped (no LLM configured)")
    
    # DSPy Binary
    print("  DSPy (binary): ", end="")
    dspy_binary = setup_dspy_reranker(score_type="binary")
    if dspy_binary:
        rerankers["DSPy-Binary"] = dspy_binary
        print("✓ Ready")
    else:
        print("✗ Skipped (no LLM configured)")
    
    print()
    
    if not rerankers:
        print("❌ No rerankers available. Please set at least one API key:")
        print("   - COHERE_API_KEY")
        print("   - VOYAGE_API_KEY")
        print("   - OPENAI_API_KEY or ANTHROPIC_API_KEY (for DSPy)")
        return
    
    print(f"✓ {len(rerankers)} reranker(s) ready: {', '.join(rerankers.keys())}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Show test data
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Test Documents")
    print("=" * 70)
    for idx, doc in enumerate(TOY_DOCUMENTS):
        preview = doc[:80].replace('\n', ' ')
        if len(doc) > 80:
            preview += "..."
        print(f"  Doc {idx}: {preview}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Run tests
    # ─────────────────────────────────────────────────────────────────────────
    for idx, query in enumerate(TOY_QUERIES):
        test_query(query, TOY_DOCUMENTS, rerankers, idx)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Expected rankings (by document relevance):")
    print("  Query 1 (RAG): Doc 0 > Doc 1 >> Doc 6, Doc 7")
    print("  Query 2 (Python): Doc 2 > Doc 3 >> Doc 6, Doc 7")
    print("  Query 3 (React): Doc 4 > Doc 5 >> Doc 6, Doc 7")


if __name__ == "__main__":
    main()

