#!/usr/bin/env python3
"""
RAG Query Result Cache

Implements LRU caching for RAG query results to avoid expensive
HyDE (Hypothetical Document Embeddings) generation on repeated queries.

Performance Impact:
- HyDE generation: 20-40s per query
- Cache hit: <0.1s (instant)
- Expected speedup: 200-400x for cached queries
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
from functools import lru_cache


class RAGQueryCache:
    """
    LRU cache for RAG query results.

    Caches both the query results (retrieved chunks) and the generated
    HyDE documents to avoid expensive LLM calls.
    """

    def __init__(self, cache_dir: Path = None, max_size: int = 100, ttl_hours: int = 24):
        """
        Initialize query cache.

        Args:
            cache_dir: Directory to store cache files (None = memory-only)
            max_size: Maximum number of queries to cache
            ttl_hours: Time-to-live for cache entries (hours)
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)

        # In-memory LRU cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []  # For LRU eviction

        # Stats
        self.hits = 0
        self.misses = 0

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_cache_key(self, query: str, n_results: int = 5) -> str:
        """Generate cache key from query and parameters."""
        cache_input = f"{query}::{n_results}"
        return hashlib.md5(cache_input.encode('utf-8')).hexdigest()

    def get(self, query: str, n_results: int = 5) -> Optional[Tuple[str, List[str]]]:
        """
        Get cached query result.

        Args:
            query: User's query
            n_results: Number of results requested

        Returns:
            (llm_response, source_notes) if cached, None otherwise
        """
        cache_key = self._make_cache_key(query, n_results)

        # Check memory cache
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]

            # Check TTL
            cached_time = datetime.fromisoformat(entry['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                # Expired
                self._evict(cache_key)
                self.misses += 1
                return None

            # Cache hit - update access order
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)

            self.hits += 1
            return (entry['response'], entry['sources'])

        # Check disk cache (if enabled)
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        entry = json.load(f)

                    # Check TTL
                    cached_time = datetime.fromisoformat(entry['timestamp'])
                    if datetime.now() - cached_time > self.ttl:
                        cache_file.unlink()
                        self.misses += 1
                        return None

                    # Load into memory cache
                    self._memory_cache[cache_key] = entry
                    self._access_order.append(cache_key)
                    self._enforce_max_size()

                    self.hits += 1
                    return (entry['response'], entry['sources'])
                except Exception:
                    # Corrupted cache file
                    cache_file.unlink(missing_ok=True)

        self.misses += 1
        return None

    def put(self, query: str, n_results: int, response: str, sources: List[str]):
        """
        Cache query result.

        Args:
            query: User's query
            n_results: Number of results requested
            response: LLM-generated response
            sources: List of source note paths
        """
        cache_key = self._make_cache_key(query, n_results)

        entry = {
            'query': query,
            'n_results': n_results,
            'response': response,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }

        # Add to memory cache
        if cache_key in self._memory_cache:
            # Update existing entry
            self._access_order.remove(cache_key)

        self._memory_cache[cache_key] = entry
        self._access_order.append(cache_key)

        # Enforce LRU eviction
        self._enforce_max_size()

        # Write to disk cache (if enabled)
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
            except Exception:
                pass  # Disk cache is optional

    def _enforce_max_size(self):
        """Evict oldest entries when cache exceeds max_size."""
        while len(self._memory_cache) > self.max_size:
            oldest_key = self._access_order.pop(0)
            self._evict(oldest_key)

    def _evict(self, cache_key: str):
        """Evict a cache entry."""
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

        if cache_key in self._access_order:
            self._access_order.remove(cache_key)

        # Delete disk cache file
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}.json"
            cache_file.unlink(missing_ok=True)

    def clear(self):
        """Clear all cache entries."""
        self._memory_cache.clear()
        self._access_order.clear()

        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()

        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'cache_size': len(self._memory_cache),
            'max_size': self.max_size
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"RAGQueryCache(size={stats['cache_size']}/{stats['max_size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.1f}%)"
        )
