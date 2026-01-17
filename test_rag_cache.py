#!/usr/bin/env python3
"""
Test RAG Query Cache

Verifies that query caching works correctly and provides expected speedup.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.services.rag_service import ObsidianRAG
from config import ProjectConfig


def test_cache_speedup():
    """Test cache hit vs cache miss performance."""
    print("\n" + "=" * 70)
    print("  TEST: RAG QUERY CACHE")
    print("=" * 70)

    # Initialize RAG
    print("\nInitializing RAG engine...")
    rag = ObsidianRAG()

    # Clear cache to start fresh
    rag.query_cache.clear()
    print(f"Cache cleared: {rag.query_cache}")

    # Test query
    test_query = "What is fibonacci?"

    # First query - CACHE MISS (expensive HyDE generation)
    print(f"\n1. First query (CACHE MISS):")
    print(f"   Query: '{test_query}'")
    start = time.time()
    try:
        results1, sources1 = rag.query(test_query, n_results=5, stream=False)
        elapsed1 = time.time() - start
        print(f"   ✅ Response time: {elapsed1:.2f}s")
        print(f"   Sources: {len(sources1)} notes")
    except Exception as e:
        elapsed1 = 0
        print(f"   ❌ Error: {e}")

    # Second query - CACHE HIT (instant)
    print(f"\n2. Second query (CACHE HIT):")
    print(f"   Query: '{test_query}'")
    start = time.time()
    try:
        results2, sources2 = rag.query(test_query, n_results=5, stream=False)
        elapsed2 = time.time() - start
        print(f"   ✅ Response time: {elapsed2:.2f}s")
        print(f"   Sources: {len(sources2)} notes")
    except Exception as e:
        elapsed2 = 0
        print(f"   ❌ Error: {e}")

    # Calculate speedup
    if elapsed1 > 0 and elapsed2 > 0:
        speedup = elapsed1 / elapsed2
        print(f"\n📊 Cache Performance:")
        print(f"  - Cache miss: {elapsed1:.2f}s")
        print(f"  - Cache hit: {elapsed2:.2f}s")
        print(f"  - Speedup: {speedup:.1f}x")

        # Verify correctness
        response1_text = results1[0]['message']['content']
        response2_text = results2[0]['message']['content']

        if response1_text == response2_text:
            print(f"  ✅ Responses are identical (cache is correct)")
        else:
            print(f"  ⚠️  Responses differ (potential cache bug)")

        # Expected speedup check
        if speedup > 50:
            print(f"  ✅ Excellent speedup (>50x)!")
        elif speedup > 10:
            print(f"  ✅ Good speedup (>10x)")
        else:
            print(f"  ⚠️  Low speedup (<10x), investigate")

    # Cache stats
    print(f"\n📈 Cache Statistics:")
    stats = rag.query_cache.get_stats()
    print(f"  - Total requests: {stats['total_requests']}")
    print(f"  - Cache hits: {stats['hits']}")
    print(f"  - Cache misses: {stats['misses']}")
    print(f"  - Hit rate: {stats['hit_rate']:.1f}%")
    print(f"  - Cache size: {stats['cache_size']}/{stats['max_size']}")

    print("\n✅ Cache test complete!")


if __name__ == "__main__":
    test_cache_speedup()
