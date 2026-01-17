"""
Spaced Repetition System (SM-2 Algorithm)

Implements SuperMemo 2 algorithm for optimal note review scheduling.
Integrates with Obsidian vault through frontmatter metadata.

References:
- SM-2 Algorithm: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2
- Paper: Wozniak, P. A. (1990). Optimization of learning
"""
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import yaml

logger = logging.getLogger("SpacedRepetition")


@dataclass
class ReviewCard:
    """
    Represents a single note's review state using SM-2 algorithm.

    Attributes:
        ease_factor: Difficulty multiplier (1.3-2.5, default 2.5)
        interval: Days until next review
        repetitions: Number of successful reviews
        next_review: Date of next scheduled review
        last_reviewed: Date of last review
        quality: Last review quality score (0-5)
    """
    ease_factor: float = 2.5
    interval: int = 1
    repetitions: int = 0
    next_review: Optional[datetime] = None
    last_reviewed: Optional[datetime] = None
    quality: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for YAML frontmatter."""
        data = asdict(self)
        # Convert datetime to string
        if self.next_review:
            data['next_review'] = self.next_review.strftime('%Y-%m-%d')
        if self.last_reviewed:
            data['last_reviewed'] = self.last_reviewed.strftime('%Y-%m-%d')
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReviewCard':
        """Load from frontmatter dictionary."""
        # Parse datetime strings
        if 'next_review' in data and isinstance(data['next_review'], str):
            data['next_review'] = datetime.strptime(data['next_review'], '%Y-%m-%d')
        if 'last_reviewed' in data and isinstance(data['last_reviewed'], str):
            data['last_reviewed'] = datetime.strptime(data['last_reviewed'], '%Y-%m-%d')

        return cls(**data)


class SM2Algorithm:
    """
    SuperMemo 2 (SM-2) spaced repetition algorithm.

    The algorithm calculates optimal review intervals based on:
    - Previous performance (quality score 0-5)
    - Ease of recall (ease_factor)
    - Number of successful repetitions
    """

    @staticmethod
    def calculate_next_review(
        card: ReviewCard,
        quality: int
    ) -> ReviewCard:
        """
        Calculate next review date based on quality score.

        Args:
            card: Current review card state
            quality: Quality of recall (0-5):
                5 - Perfect recall
                4 - Correct after hesitation
                3 - Correct with difficulty
                2 - Wrong, but recognized
                1 - Wrong, barely recognized
                0 - Complete blackout

        Returns:
            Updated ReviewCard with new interval and next_review date
        """
        if not 0 <= quality <= 5:
            raise ValueError(f"Quality must be 0-5, got {quality}")

        # Update ease factor (EF)
        card.ease_factor = card.ease_factor + (
            0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
        )

        # Ensure EF stays within bounds
        card.ease_factor = max(1.3, card.ease_factor)

        # Calculate interval
        if quality < 3:
            # Failed review - restart
            card.interval = 1
            card.repetitions = 0
        else:
            # Successful review
            if card.repetitions == 0:
                card.interval = 1
            elif card.repetitions == 1:
                card.interval = 6
            else:
                card.interval = round(card.interval * card.ease_factor)

            card.repetitions += 1

        # Set next review date
        card.last_reviewed = datetime.now()
        card.next_review = datetime.now() + timedelta(days=card.interval)
        card.quality = quality

        return card


class ObsidianReviewManager:
    """
    Manages spaced repetition for Obsidian notes.

    Features:
    - Reads/writes SM-2 metadata to note frontmatter
    - Finds notes due for review today
    - Tracks review history
    - Auto-tagging with #review tags
    """

    def __init__(self, vault_path: Path):
        """Initialize with Obsidian vault path."""
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

    def get_review_card(self, note_path: Path) -> Optional[ReviewCard]:
        """
        Extract review card from note's frontmatter.

        Args:
            note_path: Path to Obsidian note

        Returns:
            ReviewCard if note has review metadata, None otherwise
        """
        if not note_path.exists():
            return None

        content = note_path.read_text(encoding='utf-8')

        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if not frontmatter_match:
            return None

        try:
            frontmatter = yaml.safe_load(frontmatter_match.group(1))

            # Check if has review data
            if 'review' not in frontmatter:
                return None

            review_data = frontmatter['review']
            return ReviewCard.from_dict(review_data)

        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in {note_path.name}: {e}")
            return None

    def save_review_card(self, note_path: Path, card: ReviewCard):
        """
        Save review card to note's frontmatter.

        Args:
            note_path: Path to Obsidian note
            card: ReviewCard to save
        """
        if not note_path.exists():
            raise FileNotFoundError(f"Note not found: {note_path}")

        content = note_path.read_text(encoding='utf-8')

        # Extract frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---\n(.*)', content, re.DOTALL)

        if frontmatter_match:
            # Update existing frontmatter
            try:
                frontmatter = yaml.safe_load(frontmatter_match.group(1))
                body = frontmatter_match.group(2)
            except yaml.YAMLError:
                frontmatter = {}
                body = content
        else:
            # Create new frontmatter
            frontmatter = {}
            body = content

        # Update review data
        frontmatter['review'] = card.to_dict()

        # Add #review tag if not present
        tags = frontmatter.get('tags', [])
        if isinstance(tags, list) and 'review' not in tags:
            tags.append('review')
            frontmatter['tags'] = tags

        # Rebuild file
        new_content = "---\n"
        new_content += yaml.dump(frontmatter, allow_unicode=True, sort_keys=False)
        new_content += "---\n"
        new_content += body

        note_path.write_text(new_content, encoding='utf-8')
        logger.info(f"Updated review card: {note_path.name}")

    def get_due_reviews(self, target_date: Optional[datetime] = None) -> List[Path]:
        """
        Find all notes due for review on target date.

        Args:
            target_date: Date to check (default: today)

        Returns:
            List of note paths due for review
        """
        if target_date is None:
            target_date = datetime.now()

        due_notes = []

        # Scan all markdown files
        for note_path in self.vault_path.rglob("*.md"):
            # Skip templates and system folders
            if '.obsidian' in str(note_path) or 'Templates' in str(note_path):
                continue

            card = self.get_review_card(note_path)

            if card and card.next_review:
                # Check if due today or overdue
                if card.next_review.date() <= target_date.date():
                    due_notes.append(note_path)

        # Sort by next_review date (oldest first)
        due_notes.sort(
            key=lambda p: self.get_review_card(p).next_review
        )

        return due_notes

    def record_review(
        self,
        note_path: Path,
        quality: int
    ) -> ReviewCard:
        """
        Record a review and update next review date.

        Args:
            note_path: Path to reviewed note
            quality: Quality score (0-5)

        Returns:
            Updated ReviewCard
        """
        # Get current card or create new one
        card = self.get_review_card(note_path)
        if card is None:
            card = ReviewCard()
            logger.info(f"Creating new review card for {note_path.name}")

        # Calculate next review using SM-2
        card = SM2Algorithm.calculate_next_review(card, quality)

        # Save to note
        self.save_review_card(note_path, card)

        logger.info(
            f"Review recorded: {note_path.name} "
            f"(Q={quality}, next in {card.interval} days)"
        )

        return card

    def initialize_note_for_review(self, note_path: Path):
        """
        Add review metadata to a note (opt-in to review system).

        Args:
            note_path: Path to note to initialize
        """
        card = self.get_review_card(note_path)

        if card is not None:
            logger.warning(f"Note already has review card: {note_path.name}")
            return

        # Create new card with first review tomorrow
        card = ReviewCard()
        card.next_review = datetime.now() + timedelta(days=1)

        self.save_review_card(note_path, card)
        logger.info(f"Initialized review card: {note_path.name}")

    def get_review_stats(self) -> Dict:
        """
        Get review statistics across vault.

        Returns:
            Dictionary with stats:
            - total_cards: Total notes in review system
            - due_today: Notes due today
            - overdue: Notes past due
            - avg_ease_factor: Average difficulty
        """
        total = 0
        due_today = 0
        overdue = 0
        ease_factors = []

        today = datetime.now().date()

        for note_path in self.vault_path.rglob("*.md"):
            if '.obsidian' in str(note_path):
                continue

            card = self.get_review_card(note_path)
            if card and card.next_review:
                total += 1
                ease_factors.append(card.ease_factor)

                review_date = card.next_review.date()
                if review_date == today:
                    due_today += 1
                elif review_date < today:
                    overdue += 1

        return {
            'total_cards': total,
            'due_today': due_today,
            'overdue': overdue,
            'avg_ease_factor': sum(ease_factors) / len(ease_factors) if ease_factors else 0
        }


if __name__ == "__main__":
    # Example usage
    from config import ProjectConfig

    manager = ObsidianReviewManager(ProjectConfig.OBSIDIAN_VAULT)

    stats = manager.get_review_stats()
    print(f"Review Stats:")
    print(f"  Total cards: {stats['total_cards']}")
    print(f"  Due today: {stats['due_today']}")
    print(f"  Overdue: {stats['overdue']}")
    print(f"  Avg ease factor: {stats['avg_ease_factor']:.2f}")
