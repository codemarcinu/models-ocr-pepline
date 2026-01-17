#!/usr/bin/env python3
"""
Obsidian Notes Migration Script

Migrates existing notes to the new standardized format:
- Fixes YAML frontmatter escaping
- Unifies field names (date -> created)
- Updates callout syntax to modern Obsidian format
- Improves markdown structure

Usage:
    python migrate_notes.py --dry-run     # Preview changes (no modifications)
    python migrate_notes.py --migrate     # Apply changes (with backup)
    python migrate_notes.py --stats       # Show vault statistics
"""

import os
import re
import sys
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import config, but have fallback
try:
    from config import ProjectConfig
    DEFAULT_VAULT = ProjectConfig.OBSIDIAN_VAULT
except ImportError:
    DEFAULT_VAULT = Path.home() / "obsidian"

from utils.note_templates import escape_yaml_string, normalize_tag

# Rich for pretty output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich import box

console = Console()


@dataclass
class MigrationChange:
    """Represents a single change to be made."""
    file_path: Path
    change_type: str  # 'frontmatter', 'callout', 'structure'
    description: str
    before: str = ""
    after: str = ""


@dataclass
class MigrationResult:
    """Results of migration for a single file."""
    file_path: Path
    changes: List[MigrationChange] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0


class NoteMigrator:
    """Handles migration of Obsidian notes to new format."""

    # Folders to skip
    SKIP_FOLDERS = {'.obsidian', '.git', '.trash', 'node_modules', '__pycache__'}

    # Old callout patterns to modernize
    OLD_CALLOUT_PATTERNS = [
        (r'>\s*\[!INFO\]', '> [!info]'),
        (r'>\s*\[!NOTE\]', '> [!note]'),
        (r'>\s*\[!WARNING\]', '> [!warning]'),
        (r'>\s*\[!TIP\]', '> [!tip]'),
        (r'>\s*\[!DANGER\]', '> [!danger]'),
        (r'>\s*\[!ABSTRACT\]', '> [!abstract]'),
        (r'>\s*\[!SUCCESS\]', '> [!success]'),
        (r'>\s*\[!QUESTION\]', '> [!question]'),
        (r'>\s*\[!EXAMPLE\]', '> [!example]'),
        (r'>\s*\[!BUG\]', '> [!bug]'),
        (r'>\s*\[!TODO\]', '> [!todo]'),
    ]

    # Field name mappings (old -> new)
    FIELD_MAPPINGS = {
        'date': 'created',
        'Date': 'created',
        'DATE': 'created',
    }

    def __init__(self, vault_path: Path, backup_dir: Optional[Path] = None):
        self.vault_path = vault_path
        self.backup_dir = backup_dir or (vault_path / "_migration_backup")
        self.results: List[MigrationResult] = []
        self.stats = defaultdict(int)

    def scan_vault(self) -> List[Path]:
        """Scan vault for markdown files."""
        files = []
        for root, dirs, filenames in os.walk(self.vault_path):
            # Filter out skip folders
            dirs[:] = [d for d in dirs if d not in self.SKIP_FOLDERS]

            for filename in filenames:
                if filename.endswith('.md'):
                    files.append(Path(root) / filename)

        return sorted(files)

    def parse_frontmatter(self, content: str) -> Tuple[Optional[Dict], str, int, int]:
        """
        Parse YAML frontmatter from content.

        Returns:
            (frontmatter_dict, body, start_pos, end_pos)
        """
        if not content.startswith('---'):
            return None, content, 0, 0

        # Find end of frontmatter
        end_match = re.search(r'\n---\s*\n', content[3:])
        if not end_match:
            return None, content, 0, 0

        end_pos = end_match.end() + 3
        frontmatter_str = content[4:end_match.start() + 3]
        body = content[end_pos:]

        try:
            frontmatter = yaml.safe_load(frontmatter_str)
            if not isinstance(frontmatter, dict):
                return None, content, 0, 0
            return frontmatter, body, 0, end_pos
        except yaml.YAMLError:
            return None, content, 0, 0

    def fix_frontmatter(self, frontmatter: Dict) -> Tuple[Dict, List[str]]:
        """
        Fix frontmatter issues.

        Returns:
            (fixed_frontmatter, list_of_changes)
        """
        changes = []
        fixed = dict(frontmatter)

        # 1. Rename fields
        for old_name, new_name in self.FIELD_MAPPINGS.items():
            if old_name in fixed and new_name not in fixed:
                fixed[new_name] = fixed.pop(old_name)
                changes.append(f"Renamed '{old_name}' → '{new_name}'")

        # 2. Normalize tags
        if 'tags' in fixed:
            old_tags = fixed['tags']
            if isinstance(old_tags, list):
                new_tags = []
                for t in old_tags:
                    if t is None:
                        continue
                    if isinstance(t, str):
                        normalized = normalize_tag(t)
                        if normalized:
                            new_tags.append(normalized)
                    elif isinstance(t, dict):
                        # Handle nested tag dicts (e.g., {tag: value})
                        for k in t.keys():
                            if isinstance(k, str):
                                normalized = normalize_tag(k)
                                if normalized:
                                    new_tags.append(normalized)
                    else:
                        # Try to convert to string
                        try:
                            normalized = normalize_tag(str(t))
                            if normalized:
                                new_tags.append(normalized)
                        except:
                            pass
                if new_tags != old_tags:
                    fixed['tags'] = new_tags
                    changes.append(f"Normalized tags")
            elif isinstance(old_tags, str):
                # Convert string tags to list
                tags_list = [normalize_tag(t.strip()) for t in old_tags.split(',')]
                fixed['tags'] = [t for t in tags_list if t]
                changes.append(f"Converted tags string to list")

        # 3. Add missing 'type' field based on tags or content
        if 'type' not in fixed:
            tags = fixed.get('tags', [])
            if isinstance(tags, list):
                if 'paragon' in tags or 'receipt' in tags:
                    fixed['type'] = 'receipt'
                    changes.append("Added type: receipt")
                elif 'research' in tags or 'web' in tags:
                    fixed['type'] = 'research'
                    changes.append("Added type: research")
                elif 'daily' in tags:
                    fixed['type'] = 'daily'
                    changes.append("Added type: daily")
                elif 'transkrypcja' in tags or 'transcript' in tags:
                    fixed['type'] = 'transcript'
                    changes.append("Added type: transcript")

        return fixed, changes

    def build_frontmatter(self, data: Dict) -> str:
        """Build properly escaped YAML frontmatter."""
        lines = ['---']

        # Define field order for consistency
        field_order = ['title', 'created', 'summary', 'type', 'status', 'tags']

        # First add ordered fields
        for key in field_order:
            if key in data:
                lines.append(self._format_yaml_field(key, data[key]))

        # Then add remaining fields
        for key, value in data.items():
            if key not in field_order:
                lines.append(self._format_yaml_field(key, value))

        lines.append('---')
        return '\n'.join(lines)

    def _format_yaml_field(self, key: str, value: Any) -> str:
        """Format a single YAML field with proper escaping."""
        if value is None:
            return f"{key}: null"

        if isinstance(value, bool):
            return f"{key}: {str(value).lower()}"

        if isinstance(value, (int, float)):
            return f"{key}: {value}"

        if isinstance(value, list):
            if not value:
                return f"{key}: []"
            lines = [f"{key}:"]
            for item in value:
                if isinstance(item, str):
                    lines.append(f"  - {normalize_tag(item)}")
                else:
                    lines.append(f"  - {item}")
            return '\n'.join(lines)

        if isinstance(value, dict):
            lines = [f"{key}:"]
            for k, v in value.items():
                if isinstance(v, str):
                    lines.append(f"  {k}: {escape_yaml_string(v)}")
                else:
                    lines.append(f"  {k}: {v}")
            return '\n'.join(lines)

        # String value - escape if needed
        return f"{key}: {escape_yaml_string(str(value))}"

    def fix_callouts(self, content: str) -> Tuple[str, List[str]]:
        """Fix callout syntax to modern Obsidian format."""
        changes = []
        fixed = content

        for pattern, replacement in self.OLD_CALLOUT_PATTERNS:
            if re.search(pattern, fixed, re.IGNORECASE):
                fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)
                changes.append(f"Fixed callout: {replacement}")

        return fixed, changes

    def fix_structure(self, content: str) -> Tuple[str, List[str]]:
        """Fix markdown structure issues."""
        changes = []
        fixed = content

        # 1. Fix multiple consecutive blank lines (keep max 2)
        old_len = len(fixed)
        fixed = re.sub(r'\n{4,}', '\n\n\n', fixed)
        if len(fixed) != old_len:
            changes.append("Reduced excessive blank lines")

        # 2. Ensure headers have space after #
        if re.search(r'^#{1,6}[^\s#]', fixed, re.MULTILINE):
            fixed = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', fixed, flags=re.MULTILINE)
            changes.append("Fixed header spacing")

        # 3. Fix common encoding issues
        replacements = [
            ('â€"', '—'),  # em dash
            ('â€"', '–'),  # en dash
            ('â€™', "'"),  # apostrophe
            ('â€œ', '"'),  # left quote
            ('â€', '"'),   # right quote
        ]
        for old, new in replacements:
            if old in fixed:
                fixed = fixed.replace(old, new)
                changes.append(f"Fixed encoding: {old} → {new}")

        return fixed, changes

    def analyze_file(self, file_path: Path) -> MigrationResult:
        """Analyze a single file and determine needed changes."""
        result = MigrationResult(file_path=file_path)

        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding='latin-1')
            except Exception as e:
                result.error = f"Cannot read file: {e}"
                return result
        except Exception as e:
            result.error = f"Error: {e}"
            return result

        # 1. Analyze frontmatter
        frontmatter, body, fm_start, fm_end = self.parse_frontmatter(content)

        if frontmatter:
            fixed_fm, fm_changes = self.fix_frontmatter(frontmatter)
            if fm_changes:
                new_fm_str = self.build_frontmatter(fixed_fm)
                old_fm_str = content[fm_start:fm_end].strip()

                for change_desc in fm_changes:
                    result.changes.append(MigrationChange(
                        file_path=file_path,
                        change_type='frontmatter',
                        description=change_desc,
                        before=old_fm_str[:200] + '...' if len(old_fm_str) > 200 else old_fm_str,
                        after=new_fm_str[:200] + '...' if len(new_fm_str) > 200 else new_fm_str
                    ))

        # 2. Analyze callouts
        fixed_body, callout_changes = self.fix_callouts(body if frontmatter else content)
        for change_desc in callout_changes:
            result.changes.append(MigrationChange(
                file_path=file_path,
                change_type='callout',
                description=change_desc
            ))

        # 3. Analyze structure
        final_body, structure_changes = self.fix_structure(fixed_body)
        for change_desc in structure_changes:
            result.changes.append(MigrationChange(
                file_path=file_path,
                change_type='structure',
                description=change_desc
            ))

        return result

    def migrate_file(self, file_path: Path, create_backup: bool = True) -> MigrationResult:
        """Migrate a single file."""
        result = MigrationResult(file_path=file_path)

        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            content = file_path.read_text(encoding='latin-1')
        except Exception as e:
            result.error = f"Cannot read: {e}"
            return result

        original_content = content

        # 1. Fix frontmatter
        frontmatter, body, fm_start, fm_end = self.parse_frontmatter(content)

        if frontmatter:
            fixed_fm, fm_changes = self.fix_frontmatter(frontmatter)
            if fm_changes:
                new_fm_str = self.build_frontmatter(fixed_fm)
                content = new_fm_str + '\n\n' + body.lstrip()
                for desc in fm_changes:
                    result.changes.append(MigrationChange(
                        file_path=file_path,
                        change_type='frontmatter',
                        description=desc
                    ))

        # 2. Fix callouts
        content, callout_changes = self.fix_callouts(content)
        for desc in callout_changes:
            result.changes.append(MigrationChange(
                file_path=file_path,
                change_type='callout',
                description=desc
            ))

        # 3. Fix structure
        content, structure_changes = self.fix_structure(content)
        for desc in structure_changes:
            result.changes.append(MigrationChange(
                file_path=file_path,
                change_type='structure',
                description=desc
            ))

        # Only write if changes were made
        if content != original_content:
            if create_backup:
                self._create_backup(file_path, original_content)

            file_path.write_text(content, encoding='utf-8')

        return result

    def _create_backup(self, file_path: Path, content: str):
        """Create backup of original file."""
        relative = file_path.relative_to(self.vault_path)
        backup_path = self.backup_dir / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(content, encoding='utf-8')

    def get_stats(self) -> Dict[str, int]:
        """Get vault statistics."""
        stats = {
            'total_files': 0,
            'with_frontmatter': 0,
            'without_frontmatter': 0,
            'needs_migration': 0,
            'by_type': defaultdict(int),
            'by_folder': defaultdict(int),
        }

        files = self.scan_vault()
        stats['total_files'] = len(files)

        for file_path in files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                frontmatter, _, _, _ = self.parse_frontmatter(content)

                if frontmatter:
                    stats['with_frontmatter'] += 1
                    note_type = frontmatter.get('type', 'unknown')
                    stats['by_type'][note_type] += 1
                else:
                    stats['without_frontmatter'] += 1

                # Check if needs migration
                result = self.analyze_file(file_path)
                if result.has_changes:
                    stats['needs_migration'] += 1

                # Count by folder
                folder = file_path.parent.relative_to(self.vault_path).parts[0] if file_path.parent != self.vault_path else '/'
                stats['by_folder'][folder] += 1

            except Exception:
                continue

        return stats


def show_stats(migrator: NoteMigrator):
    """Display vault statistics."""
    console.print("\n[bold cyan]📊 Skanowanie vault...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Analizowanie...", total=None)
        stats = migrator.get_stats()

    # Summary table
    table = Table(title="📁 Statystyki Vault", box=box.ROUNDED)
    table.add_column("Metryka", style="cyan")
    table.add_column("Wartość", justify="right", style="green")

    table.add_row("Wszystkie pliki .md", str(stats['total_files']))
    table.add_row("Z frontmatter", str(stats['with_frontmatter']))
    table.add_row("Bez frontmatter", str(stats['without_frontmatter']))
    table.add_row("[yellow]Wymaga migracji[/]", f"[yellow]{stats['needs_migration']}[/]")

    console.print(table)

    # By type
    if stats['by_type']:
        type_table = Table(title="📝 Notatki wg typu", box=box.SIMPLE)
        type_table.add_column("Typ", style="cyan")
        type_table.add_column("Ilość", justify="right")

        for note_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
            type_table.add_row(note_type, str(count))

        console.print(type_table)

    # By folder (top 10)
    if stats['by_folder']:
        folder_table = Table(title="📂 Notatki wg folderu (top 10)", box=box.SIMPLE)
        folder_table.add_column("Folder", style="cyan")
        folder_table.add_column("Ilość", justify="right")

        for folder, count in sorted(stats['by_folder'].items(), key=lambda x: -x[1])[:10]:
            folder_table.add_row(folder, str(count))

        console.print(folder_table)


def dry_run(migrator: NoteMigrator):
    """Preview changes without modifying files."""
    console.print("\n[bold cyan]🔍 Dry Run - Podgląd zmian[/]\n")

    files = migrator.scan_vault()
    changes_by_type = defaultdict(list)
    files_with_changes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Analizowanie plików...", total=len(files))

        for file_path in files:
            result = migrator.analyze_file(file_path)

            if result.has_changes:
                files_with_changes.append(result)
                for change in result.changes:
                    changes_by_type[change.change_type].append(change)

            progress.advance(task)

    # Summary
    console.print(Panel(
        f"[green]Plików do zmiany:[/] {len(files_with_changes)} / {len(files)}",
        title="📋 Podsumowanie Dry Run"
    ))

    # Changes by type
    if changes_by_type:
        table = Table(title="🔧 Zmiany wg typu", box=box.ROUNDED)
        table.add_column("Typ zmiany", style="cyan")
        table.add_column("Ilość", justify="right", style="yellow")

        for change_type, changes in sorted(changes_by_type.items()):
            table.add_row(change_type, str(len(changes)))

        console.print(table)

    # Show sample changes (first 5 files)
    if files_with_changes:
        console.print("\n[bold]📄 Przykładowe zmiany (pierwsze 5 plików):[/]\n")

        for result in files_with_changes[:5]:
            rel_path = result.file_path.relative_to(migrator.vault_path)
            console.print(f"[cyan]📝 {rel_path}[/]")

            for change in result.changes:
                console.print(f"   • [yellow]{change.change_type}[/]: {change.description}")

            console.print()

    if files_with_changes:
        console.print(f"\n[bold green]✓[/] Uruchom z [cyan]--migrate[/] aby zastosować zmiany")
        console.print(f"  Backup zostanie utworzony w: [dim]{migrator.backup_dir}[/]")
    else:
        console.print("\n[bold green]✓[/] Brak plików wymagających migracji!")


def run_migration(migrator: NoteMigrator):
    """Run the actual migration."""
    console.print("\n[bold cyan]🚀 Rozpoczynam migrację...[/]\n")

    # First do a dry run to count
    files = migrator.scan_vault()
    files_to_migrate = []

    for file_path in files:
        result = migrator.analyze_file(file_path)
        if result.has_changes:
            files_to_migrate.append(file_path)

    if not files_to_migrate:
        console.print("[green]✓ Brak plików do migracji![/]")
        return

    console.print(f"[yellow]⚠️  Migracja {len(files_to_migrate)} plików[/]")
    console.print(f"[dim]Backup: {migrator.backup_dir}[/]\n")

    # Create backup directory
    migrator.backup_dir.mkdir(parents=True, exist_ok=True)

    migrated = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Migrowanie...", total=len(files_to_migrate))

        for file_path in files_to_migrate:
            result = migrator.migrate_file(file_path, create_backup=True)

            if result.error:
                errors += 1
                console.print(f"[red]✗[/] {file_path.name}: {result.error}")
            elif result.has_changes:
                migrated += 1

            progress.advance(task)

    # Summary
    console.print()
    console.print(Panel(
        f"[green]Zmigrowano:[/] {migrated}\n"
        f"[red]Błędy:[/] {errors}\n"
        f"[dim]Backup:[/] {migrator.backup_dir}",
        title="✅ Migracja zakończona"
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Obsidian notes to new format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--migrate', action='store_true',
                        help='Apply migration (creates backup)')
    parser.add_argument('--stats', action='store_true',
                        help='Show vault statistics')
    parser.add_argument('--vault', type=str, default=None,
                        help='Path to Obsidian vault')

    args = parser.parse_args()

    vault_path = Path(args.vault) if args.vault else DEFAULT_VAULT

    if not vault_path.exists():
        console.print(f"[red]❌ Vault nie istnieje:[/] {vault_path}")
        return 1

    console.print(Panel(
        f"[cyan]Vault:[/] {vault_path}",
        title="🧠 Obsidian Notes Migrator"
    ))

    migrator = NoteMigrator(vault_path)

    if args.stats:
        show_stats(migrator)
    elif args.dry_run:
        dry_run(migrator)
    elif args.migrate:
        run_migration(migrator)
    else:
        # Default: show stats and dry run
        show_stats(migrator)
        dry_run(migrator)

    return 0


if __name__ == "__main__":
    sys.exit(main())
