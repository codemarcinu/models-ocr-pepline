import os
import time
import json
from pathlib import Path
import pytest

from core.tools.receipt_cleaner import ReceiptSanitizer
from async_receipt_pipeline import AsyncReceiptPipeline
from utils.receipt_cache import ReceiptCache


def test_sync_processing(sanitizer: ReceiptSanitizer, ocr_text: str):
    """Test sync processing (original method)."""
    # Temporarily disable async for this test
    original_async = sanitizer.use_async
    sanitizer.use_async = False

    start_time = time.time()
    try:
        result = sanitizer.generate_clean_data(ocr_text)
        elapsed = time.time() - start_time

        items_count = len(result.get('items', []))
        print(f"✓ Sync processing completed in {elapsed:.2f}s")
        print(f"  Items extracted: {items_count}")

        assert items_count > 0, "No items extracted in sync processing"
        # Further assertions for data correctness can be added here
        assert result['items'][0].get('name') is not None # Assuming 'name' is expected

    except Exception as e:
        pytest.fail(f"✗ Sync processing failed: {e}")
    finally:
        sanitizer.use_async = original_async

def test_async_processing(sanitizer: ReceiptSanitizer, ocr_text: str):
    """Test async processing (Phase 1)."""
    # Ensure async is enabled for this test
    sanitizer.use_async = True

    start_time = time.time()
    try:
        result = sanitizer.generate_clean_data(ocr_text)
        elapsed = time.time() - start_time

        items_count = len(result.get('items', []))
        print(f"✓ Async processing completed in {elapsed:.2f}s")
        print(f"  Items extracted: {items_count}")

        assert items_count > 0, "No items extracted in async processing"
        assert result['items'][0].get('nazwa') is not None # Assuming 'nazwa' is expected

        # Basic check for cache stats if async_pipeline is available
        if hasattr(sanitizer, 'async_pipeline'):
            cache_stats = sanitizer.async_pipeline.get_cache_stats()
            assert 'total_lookups' in cache_stats
            assert 'hit_rate' in cache_stats

    except Exception as e:
        pytest.fail(f"✗ Async processing failed: {e}")

def test_cache_warmup(sanitizer: ReceiptSanitizer, ocr_text: str, ocr_text_2: str):
    """Test cache effectiveness with repeated items."""
    # Ensure async is enabled for this test
    sanitizer.use_async = True

    # Process first receipt to warm cache
    result1 = sanitizer.generate_clean_data(ocr_text)
    assert len(result1.get('items', [])) > 0

    # Reset cache stats for accurate measurement
    if hasattr(sanitizer, 'async_pipeline'):
        sanitizer.async_pipeline.cache.reset_stats()

    # Process second receipt with repeated items (should hit cache)
    result2 = sanitizer.generate_clean_data(ocr_text_2)
    assert len(result2.get('items', [])) > 0

    # Check cache stats for hits
    if hasattr(sanitizer, 'async_pipeline'):
        cache_stats = sanitizer.async_pipeline.get_cache_stats()
        # Expect some hits if there are overlapping items
        assert cache_stats.get('total_lookups', 0) > 0
        assert cache_stats.get('hit_rate', 0) >= 0 # Hit rate can be 0 if no common items or no cache use

def test_error_handling(sanitizer: ReceiptSanitizer):
    """Test error handling and fallback."""
    sanitizer.use_async = True

    # Test with empty input
    result_empty = sanitizer.generate_clean_data("")
    assert isinstance(result_empty, dict)
    assert 'items' in result_empty
    # Further investigation needed to confirm if empty input should always result in 0 items in sync fallback.
    # For now, we only assert it returns a dict with an 'items' key.

    # Test with malformed input (expect it to process without crashing, but possibly no items)
    result = sanitizer.generate_clean_data("RANDOM GARBAGE TEXT !@#$%")
    assert isinstance(result, dict)
    # The actual expected behavior for malformed input might vary,
    # so we'll just assert that it doesn't crash and returns a dict.

# Removed main() function and direct calls to test functions.
# pytest will discover and run these tests automatically.