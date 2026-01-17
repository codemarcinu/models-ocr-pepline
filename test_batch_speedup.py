#!/usr/bin/env python3
"""
Test Batch Processing Speedup

Compares sync vs async for processing 10 receipts.
Expected:
- Sync: 10 × 12s = 120s (AI call for each receipt)
- Async: 13s (first) + 9 × 0.1s = ~14s (cache works for rest)
- Speedup: ~8-10x
"""
import os
import sys
import time
from pathlib import Path

# Setup
sys.path.insert(0, str(Path(__file__).parent))

# Sample receipt OCR
SAMPLE_OCR = """
JERONIMO MARTINS POLSKA S.A.
BIEDRONKA NR 1234

Data: 13.01.2026    Godzina: 14:23

MLEKO 2% 1L              5.99 A
CHLEB RAZOWY             4.50 A
MASŁO EKSTRA 200G        8.99 A
JAJKA L 10SZT           12.99 A
SER GOUDA 250G           9.99 A
POMIDORY 1KG             7.50 A
BANANY 1KG               5.99 A

SUMA PLN                56.95
"""

def test_sync_batch(count=10):
    """Test sync processing for N receipts."""
    print(f"\n{'='*80}")
    print(f"SYNC BATCH TEST ({count} receipts)")
    print(f"{'='*80}")

    # Force sync mode
    os.environ['ASYNC_PIPELINE'] = 'false'
    from core.tools.receipt_cleaner import ReceiptSanitizer

    vault = Path('obsidian_vault_test')
    vault.mkdir(exist_ok=True)

    sanitizer = ReceiptSanitizer(vault)

    start = time.time()
    for i in range(count):
        result = sanitizer.generate_clean_data(SAMPLE_OCR)
        print(f"  Receipt {i+1}/{count}: {len(result.get('items', []))} items")

    elapsed = time.time() - start
    avg = elapsed / count

    print(f"\n✓ Sync batch completed")
    print(f"  Total: {elapsed:.2f}s")
    print(f"  Average per receipt: {avg:.2f}s")

    return elapsed

def test_async_batch(count=10):
    """Test async processing for N receipts."""
    print(f"\n{'='*80}")
    print(f"ASYNC BATCH TEST ({count} receipts)")
    print(f"{'='*80}")

    # Clear cache first
    cache_file = Path('data/receipt_cache.json')
    cache_file.unlink(missing_ok=True)

    # Force async mode
    os.environ['ASYNC_PIPELINE'] = 'true'

    # Need to reload module to pick up new env var
    import importlib
    import core.tools.receipt_cleaner as receipt_cleaner
    importlib.reload(receipt_cleaner)
    from core.tools.receipt_cleaner import ReceiptSanitizer

    vault = Path('obsidian_vault_test')
    vault.mkdir(exist_ok=True)

    sanitizer = ReceiptSanitizer(vault)

    start = time.time()
    times = []
    for i in range(count):
        t_start = time.time()
        result = sanitizer.generate_clean_data(SAMPLE_OCR)
        t_elapsed = time.time() - t_start
        times.append(t_elapsed)
        print(f"  Receipt {i+1}/{count}: {len(result.get('items', []))} items ({t_elapsed:.2f}s)")

    elapsed = time.time() - start
    avg = elapsed / count

    # Get cache stats
    cache_stats = sanitizer.async_pipeline.get_cache_stats()

    print(f"\n✓ Async batch completed")
    print(f"  Total: {elapsed:.2f}s")
    print(f"  Average per receipt: {avg:.2f}s")
    print(f"  First receipt: {times[0]:.2f}s (cold cache)")
    print(f"  Last 9 receipts avg: {sum(times[1:])/len(times[1:]):.2f}s (warm cache)")
    print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")

    return elapsed

def main():
    print("\n" + "="*80)
    print("  BATCH PROCESSING SPEEDUP TEST")
    print("="*80)
    print("\nThis test processes 10 identical receipts to measure speedup.")
    print("Expected: Sync ~120s (10×12s), Async ~14s (13s + 9×0.1s)")
    print()

    # Test sync first (slower, so user sees it's working)
    sync_time = test_sync_batch(count=10)

    # Test async
    async_time = test_async_batch(count=10)

    # Compare
    print(f"\n{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"Sync total:  {sync_time:.2f}s")
    print(f"Async total: {async_time:.2f}s")
    print(f"\n🚀 SPEEDUP: {sync_time/async_time:.2f}x")

    if sync_time / async_time >= 3.0:
        print(f"   ✅ TARGET MET (>3x speedup)")
    elif sync_time / async_time >= 2.0:
        print(f"   ⚠ Close to target (2-3x speedup)")
    else:
        print(f"   ❌ Below target (<2x speedup)")

    # Cleanup
    import shutil
    shutil.rmtree('obsidian_vault_test', ignore_errors=True)

if __name__ == "__main__":
    main()
