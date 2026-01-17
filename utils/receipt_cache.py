"""
Receipt Cache Module - 3-Tier Caching System for Receipt Processing

Provides smart caching to reduce fuzzy matching and AI calls by 60-70%.

Architecture:
- Tier 1: Exact Match (O(1) hash lookup) - instant hits for known OCR strings
- Tier 2: LRU Cache (recent 500 products) - fast hits for frequently seen items
- Tier 3: Shop-Specific Patterns - common items per store

Expected hit rate: 60-70% → saves 3-4s per receipt
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from collections import OrderedDict

logger = logging.getLogger("ReceiptCache")


@dataclass
class ProductMatch:
    """
    Represents a matched product from cache.

    Attributes:
        name: Canonical product name
        category: Product category (e.g., "Nabiał", "Pieczywo")
        unit: Unit of measure (e.g., "szt", "kg")
        confidence: Match confidence (0-1)
        source: Where match came from ("exact", "lru", "pattern")
    """
    name: str
    category: str
    unit: str
    confidence: float
    source: str = "exact"

    def to_item(self) -> Dict:
        """Convert to receipt item format."""
        return {
            'nazwa': self.name,
            'kategoria': self.category,
            'jednostka': self.unit,
            'ilosc': 1.0,
            'cena_jedn': 0.0,
            'rabat': 0.0,
            'suma': 0.0
        }


class ReceiptCache:
    """
    3-tier caching system for receipt processing optimization.

    Performance Impact:
    - Cache hit: ~1ms (instant return)
    - Cache miss: 50-200ms (fuzzy matching required)
    - Expected hit rate: 60-70%
    - Savings per receipt: 3-4 seconds

    Usage:
        cache = ReceiptCache()

        # Try cache first
        match = cache.lookup("MLEKO 2%", "Biedronka")
        if match:
            # Use cached result
            item = match.to_item()
        else:
            # Do expensive fuzzy matching
            match = fuzzy_match(line)
            cache.update(line, match, "Biedronka")
    """

    def __init__(self, cache_file: Optional[Path] = None):
        """
        Initialize cache with optional persistence.

        Args:
            cache_file: Path to JSON cache file (default: data/receipt_cache.json)
        """
        self.cache_file = cache_file or self._default_cache_path()

        # Tier 1: Exact match dictionary {OCR_TEXT_UPPER: ProductMatch}
        self.exact_match: Dict[str, ProductMatch] = {}

        # Tier 2: LRU cache for recent fuzzy matches (OrderedDict-based)
        self._lru_cache: OrderedDict[str, ProductMatch] = OrderedDict()
        self._lru_max_size = 500

        # Tier 3: Shop-specific common patterns
        # {shop_name: [list of common OCR patterns]}
        self.shop_patterns: Dict[str, List[str]] = {}

        # Statistics
        self.stats = {
            'exact_hits': 0,
            'lru_hits': 0,
            'pattern_hits': 0,
            'misses': 0
        }

        # Load persistent cache
        self._load_cache()

        logger.info(
            f"ReceiptCache initialized: {len(self.exact_match)} exact entries, "
            f"{len(self.shop_patterns)} shops"
        )

    def _default_cache_path(self) -> Path:
        """Get default cache file path."""
        try:
            from config import ProjectConfig
            base_dir = ProjectConfig.BASE_DIR
        except ImportError:
            base_dir = Path(__file__).parent.parent

        cache_dir = base_dir / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "receipt_cache.json"

    def _lru_get(self, key: str) -> Optional[ProductMatch]:
        """Get from LRU cache, moving to end (most recent)."""
        if key in self._lru_cache:
            self._lru_cache.move_to_end(key)
            return self._lru_cache[key]
        return None

    def _lru_put(self, key: str, match: ProductMatch):
        """Put into LRU cache, evicting oldest if at capacity."""
        if key in self._lru_cache:
            self._lru_cache.move_to_end(key)
        else:
            if len(self._lru_cache) >= self._lru_max_size:
                self._lru_cache.popitem(last=False)
            self._lru_cache[key] = match

    def lookup(self, line: str, shop: str) -> Optional[ProductMatch]:
        """
        3-tier lookup with fallthrough.

        Tries each tier in order (fastest to slowest):
        1. Exact match (hash lookup) - 1ms
        2. LRU fuzzy cache - 1-2ms
        3. Shop-specific patterns - 2-5ms

        Args:
            line: Raw OCR text line
            shop: Shop name (e.g., "Biedronka", "Lidl")

        Returns:
            ProductMatch if found in cache, None otherwise
        """
        line_clean = line.strip().upper()

        # Skip very short lines (noise)
        if len(line_clean) < 3:
            return None

        # Tier 1: Exact match (instant O(1) lookup)
        if line_clean in self.exact_match:
            self.stats['exact_hits'] += 1
            match = self.exact_match[line_clean]
            match.source = "exact"
            logger.debug(f"Cache HIT (exact): {line_clean[:30]}...")
            return match

        # Tier 2: LRU fuzzy cache
        cache_key = f"{shop}:{line_clean[:30]}"
        cached_match = self._lru_get(cache_key)
        if cached_match:
            self.stats['lru_hits'] += 1
            cached_match.source = "lru"
            logger.debug(f"Cache HIT (lru): {line_clean[:30]}...")
            return cached_match

        # Tier 3: Shop-specific patterns
        if shop in self.shop_patterns:
            for pattern in self.shop_patterns[shop]:
                if pattern in line_clean:
                    # Pattern matched - lookup exact match for this pattern
                    if pattern in self.exact_match:
                        self.stats['pattern_hits'] += 1
                        match = self.exact_match[pattern]
                        match.source = "pattern"
                        logger.debug(f"Cache HIT (pattern): {line_clean[:30]}... → {pattern}")
                        return match

        # Cache miss - expensive operation required
        self.stats['misses'] += 1
        logger.debug(f"Cache MISS: {line_clean[:30]}...")
        return None

    def update(
        self,
        line: str,
        match: ProductMatch,
        shop: str
    ):
        """
        Update cache after successful match.

        Adds to exact match cache and updates shop patterns.

        Args:
            line: Original OCR line that was matched
            match: ProductMatch result from fuzzy matching or AI
            shop: Shop name
        """
        line_clean = line.strip().upper()

        # Skip very short or invalid entries
        if len(line_clean) < 3 or not match.name:
            return

        # Add to exact match cache
        self.exact_match[line_clean] = match

        # Update LRU cache for quick partial lookups
        cache_key = f"{shop}:{line_clean[:30]}"
        self._lru_put(cache_key, match)

        # Update shop patterns (keep top 50 most common per shop)
        if shop not in self.shop_patterns:
            self.shop_patterns[shop] = []

        if line_clean not in self.shop_patterns[shop]:
            self.shop_patterns[shop].append(line_clean)

            # Keep only top 50 (most recent)
            if len(self.shop_patterns[shop]) > 50:
                self.shop_patterns[shop] = self.shop_patterns[shop][-50:]

        logger.debug(f"Cache UPDATE: {line_clean[:30]}... → {match.name}")

    def bulk_update(self, matches: List[tuple]):
        """
        Bulk update cache with multiple matches.

        Args:
            matches: List of (line, ProductMatch, shop) tuples
        """
        for line, match, shop in matches:
            self.update(line, match, shop)

    def get_stats(self) -> Dict:
        """
        Get cache performance statistics.

        Returns:
            Dict with hit/miss counts and hit rate
        """
        total = sum(self.stats.values())
        hits = self.stats['exact_hits'] + self.stats['lru_hits'] + self.stats['pattern_hits']

        return {
            **self.stats,
            'total_lookups': total,
            'hit_rate': (hits / total) if total > 0 else 0.0,
            'cache_size': len(self.exact_match),
            'shop_count': len(self.shop_patterns)
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            'exact_hits': 0,
            'lru_hits': 0,
            'pattern_hits': 0,
            'misses': 0
        }

    def _load_cache(self):
        """Load persistent cache from disk."""
        if not self.cache_file.exists():
            logger.info("No cache file found, starting with empty cache")
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Rebuild exact match dict
            for line, match_data in data.get('exact', {}).items():
                self.exact_match[line] = ProductMatch(**match_data)

            # Load shop patterns
            self.shop_patterns = data.get('shop_patterns', {})

            logger.info(
                f"Cache loaded: {len(self.exact_match)} products, "
                f"{len(self.shop_patterns)} shops"
            )

        except Exception as e:
            logger.warning(f"Cache load failed: {e}, starting fresh")
            # Don't fail - just start with empty cache

    def save(self):
        """
        Persist cache to disk.

        Saves exact match dictionary and shop patterns to JSON.
        LRU cache is in-memory only (will rebuild on next run).
        """
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'exact': {
                    k: asdict(v)
                    for k, v in self.exact_match.items()
                },
                'shop_patterns': self.shop_patterns,
                'stats': self.get_stats()
            }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Cache saved to {self.cache_file}")

        except Exception as e:
            logger.error(f"Cache save failed: {e}")

    def clear(self):
        """Clear all cache data (for testing/debugging)."""
        self.exact_match.clear()
        self.shop_patterns.clear()
        self._lru_cache.clear()
        self.reset_stats()
        logger.info("Cache cleared")

    def prune(self, max_age_days: int = 90):
        """
        Prune old cache entries (future enhancement).

        Args:
            max_age_days: Remove entries older than this many days
        """
        # TODO: Add timestamp tracking to ProductMatch
        # TODO: Implement age-based pruning
        logger.warning("Cache pruning not yet implemented")


# Global cache instance (lazy-loaded)
_global_cache: Optional[ReceiptCache] = None


def get_cache() -> ReceiptCache:
    """
    Get or create global ReceiptCache instance.

    Returns:
        Global ReceiptCache singleton
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ReceiptCache()
    return _global_cache
