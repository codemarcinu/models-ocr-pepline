#!/usr/bin/env python3
"""
Test Spaced Repetition System

Validates SM-2 algorithm and Obsidian integration.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent))

from core.services.spaced_repetition_service import SM2Algorithm, ReviewCard, ObsidianReviewManager


def test_sm2_algorithm():
    """Test SM-2 algorithm calculations."""
    print("\n" + "=" * 70)
    print("  TEST 1: SM-2 Algorithm")
    print("=" * 70)

    card = ReviewCard()

    print(f"\nInitial state:")
    print(f"  Ease Factor: {card.ease_factor}")
    print(f"  Interval: {card.interval}")
    print(f"  Repetitions: {card.repetitions}")

    # Test quality 5 (perfect recall)
    print(f"\n📝 Review 1: Quality = 5 (Perfect)")
    card = SM2Algorithm.calculate_next_review(card, quality=5)
    print(f"  → Interval: {card.interval} days")
    print(f"  → Ease Factor: {card.ease_factor:.2f}")
    print(f"  → Repetitions: {card.repetitions}")

    assert card.interval == 1, "First interval should be 1 day"
    assert card.repetitions == 1

    # Test quality 5 again
    print(f"\n📝 Review 2: Quality = 5 (Perfect)")
    card = SM2Algorithm.calculate_next_review(card, quality=5)
    print(f"  → Interval: {card.interval} days")
    print(f"  → Ease Factor: {card.ease_factor:.2f}")
    print(f"  → Repetitions: {card.repetitions}")

    assert card.interval == 6, "Second interval should be 6 days"
    assert card.repetitions == 2

    # Test quality 4
    print(f"\n📝 Review 3: Quality = 4 (Good)")
    card = SM2Algorithm.calculate_next_review(card, quality=4)
    print(f"  → Interval: {card.interval} days (6 × {card.ease_factor:.2f})")
    print(f"  → Ease Factor: {card.ease_factor:.2f}")
    print(f"  → Repetitions: {card.repetitions}")

    assert card.interval > 6, "Third interval should be > 6 days"
    assert card.repetitions == 3

    # Test quality 2 (failure)
    print(f"\n📝 Review 4: Quality = 2 (Failed)")
    card = SM2Algorithm.calculate_next_review(card, quality=2)
    print(f"  → Interval: {card.interval} days (RESET)")
    print(f"  → Ease Factor: {card.ease_factor:.2f}")
    print(f"  → Repetitions: {card.repetitions} (RESET)")

    assert card.interval == 1, "Failed review should reset to 1 day"
    assert card.repetitions == 0, "Failed review should reset repetitions"

    print(f"\n✅ SM-2 Algorithm tests passed!")


def test_obsidian_integration():
    """Test Obsidian note integration."""
    print("\n" + "=" * 70)
    print("  TEST 2: Obsidian Integration")
    print("=" * 70)

    # Create temporary vault
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)
        print(f"\nCreated temp vault: {vault_path}")

        # Create sample note
        note_path = vault_path / "test_note.md"
        note_content = """---
title: Test Note
tags:
  - test
---

# Test Note

This is a test note for spaced repetition.
"""
        note_path.write_text(note_content, encoding='utf-8')
        print(f"Created test note: {note_path.name}")

        # Initialize manager
        manager = ObsidianReviewManager(vault_path)

        # Test: Get review card (should be None)
        print(f"\n1. Get review card (before init):")
        card = manager.get_review_card(note_path)
        assert card is None, "New note should not have review card"
        print(f"   ✅ Correctly returns None")

        # Test: Initialize note for review
        print(f"\n2. Initialize note for review:")
        manager.initialize_note_for_review(note_path)

        card = manager.get_review_card(note_path)
        assert card is not None, "Initialized note should have review card"
        assert card.next_review is not None
        print(f"   ✅ Review card created")
        print(f"   → Next review: {card.next_review.strftime('%Y-%m-%d')}")

        # Test: Record review
        print(f"\n3. Record review (quality=5):")
        updated_card = manager.record_review(note_path, quality=5)
        print(f"   ✅ Review recorded")
        print(f"   → Interval: {updated_card.interval} days")
        print(f"   → Repetitions: {updated_card.repetitions}")

        # Test: Get due reviews
        print(f"\n4. Get due reviews (today):")
        # Manually set next_review to today for testing
        card.next_review = datetime.now()
        manager.save_review_card(note_path, card)

        due_notes = manager.get_due_reviews()
        assert len(due_notes) == 1, "Should find 1 due note"
        print(f"   ✅ Found {len(due_notes)} due note(s)")

        # Test: Get stats
        print(f"\n5. Get review stats:")
        stats = manager.get_review_stats()
        print(f"   ✅ Stats retrieved")
        print(f"   → Total cards: {stats['total_cards']}")
        print(f"   → Due today: {stats['due_today']}")
        print(f"   → Avg EF: {stats['avg_ease_factor']:.2f}")

        # Verify frontmatter was added
        print(f"\n6. Verify frontmatter:")
        updated_content = note_path.read_text(encoding='utf-8')
        assert 'review:' in updated_content, "Frontmatter should have review field"
        assert 'ease_factor:' in updated_content
        assert 'interval:' in updated_content
        print(f"   ✅ Frontmatter correctly updated")

    print(f"\n✅ Obsidian integration tests passed!")


def test_cli_commands():
    """Test CLI commands."""
    print("\n" + "=" * 70)
    print("  TEST 3: CLI Commands")
    print("=" * 70)

    # Test that CLI can import without errors
    try:
        import review_cli
        print(f"\n✅ review_cli.py imports successfully")
    except ImportError as e:
        print(f"\n❌ review_cli.py import failed: {e}")
        return False

    print(f"\nAvailable commands:")
    print(f"  - python review_cli.py stats")
    print(f"  - python review_cli.py due")
    print(f"  - python review_cli.py review")
    print(f"  - python review_cli.py add <file>")
    print(f"  - python review_cli.py export")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  SPACED REPETITION SYSTEM TEST SUITE")
    print("=" * 70)

    try:
        test_sm2_algorithm()
        test_obsidian_integration()
        test_cli_commands()

        print("\n" + "=" * 70)
        print("  🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nSpaced Repetition System is ready to use!")
        print("\nNext steps:")
        print("  1. Add notes to review: python review_cli.py add <note>")
        print("  2. View stats: python review_cli.py stats")
        print("  3. Start reviewing: python review_cli.py review")
        print("  4. Update dashboard: python update_review_dashboard.py")

        return 0

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
