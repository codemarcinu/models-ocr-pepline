#!/usr/bin/env python3
"""
Test Parallel Indexing Performance

Compares sequential vs parallel indexing to verify speedup.
"""
import sys
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from core.services.rag_service import ObsidianRAG


def create_test_vault(num_files: int = 20) -> Path:
    """Create temporary vault with test notes."""
    tmpdir = Path(tempfile.mkdtemp(prefix="rag_parallel_"))

    for i in range(num_files):
        note_path = tmpdir / f"test_note_{i:03d}.md"

        content = f"""---
title: Test Note {i}
tags:
  - test
  - parallel
date: {datetime.now().strftime('%Y-%m-%d')}
---

# Test Note {i}

## Introduction

This is a test note for parallel indexing benchmarks.

## Content Section 1

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua.

### Subsection {i}

More detailed content for note {i} to create multiple chunks.

## Content Section 2

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
ut aliquip ex ea commodo consequat.

## Conclusion

Summary of test note {i}.
"""
        note_path.write_text(content, encoding='utf-8')

    print(f"Created {num_files} test notes in {tmpdir}")
    return tmpdir


def test_parallel_speedup():
    """Test sequential vs parallel indexing."""
    print("\n" + "=" * 70)
    print("  TEST: PARALLEL INDEXING SPEEDUP")
    print("=" * 70)

    # Create test vault
    print("\n📝 Creating test vault...")
    vault_path = create_test_vault(num_files=50)

    try:
        # Test 1: Sequential indexing
        print("\n1. Sequential Indexing:")
        with tempfile.TemporaryDirectory(prefix="rag_db_seq_") as tmpdb:
            rag_seq = ObsidianRAG(db_path=Path(tmpdb))

            start = time.time()
            chunks_seq = rag_seq.index_vault(vault_path, parallel=False)
            time_seq = time.time() - start

            print(f"   ✅ Indexed {chunks_seq} chunks in {time_seq:.2f}s")
            print(f"   Speed: {chunks_seq / time_seq:.1f} chunks/sec")

        # Test 2: Parallel indexing (4 workers)
        print("\n2. Parallel Indexing (4 workers):")
        with tempfile.TemporaryDirectory(prefix="rag_db_par_") as tmpdb:
            rag_par = ObsidianRAG(db_path=Path(tmpdb))

            start = time.time()
            chunks_par = rag_par.index_vault(vault_path, parallel=True, max_workers=4)
            time_par = time.time() - start

            print(f"   ✅ Indexed {chunks_par} chunks in {time_par:.2f}s")
            print(f"   Speed: {chunks_par / time_par:.1f} chunks/sec")

        # Calculate speedup
        if time_seq > 0 and time_par > 0:
            speedup = time_seq / time_par
            print(f"\n📊 Parallel Processing Results:")
            print(f"  - Sequential: {time_seq:.2f}s")
            print(f"  - Parallel (4 workers): {time_par:.2f}s")
            print(f"  - Speedup: {speedup:.2f}x")

            # Verify correctness
            if chunks_seq == chunks_par:
                print(f"  ✅ Chunk count matches (correctness verified)")
            else:
                print(f"  ⚠️  Chunk count mismatch! Seq: {chunks_seq}, Par: {chunks_par}")

            # Performance check
            if speedup > 2.5:
                print(f"  ✅ Excellent speedup (>2.5x)!")
            elif speedup > 1.5:
                print(f"  ✅ Good speedup (>1.5x)")
            else:
                print(f"  ⚠️  Low speedup (<1.5x), investigate")

    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up test vault...")
        shutil.rmtree(vault_path, ignore_errors=True)

    print("\n✅ Parallel indexing test complete!")


if __name__ == "__main__":
    test_parallel_speedup()
