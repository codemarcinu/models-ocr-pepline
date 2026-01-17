import os
import datetime
import shutil
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from flashtext import KeywordProcessor

from config import ProjectConfig, logger
from utils.tag_engine import TagEngine
from utils.note_templates import (
    escape_yaml_string, normalize_tag, sanitize_filename,
    NoteBuilder, DailyNoteBuilder, TranscriptNoteBuilder, create_dataview_query
)
# Import RAG ONLY if needed to avoid circular imports during init
from core.services.rag_service import ObsidianRAG

class LinkOptimizer:
    """
    High-performance keyword replacement using FlashText (Aho-Corasick algorithm).
    Replaces O(N*M) regex searches with O(N) single-pass scan.

    Supports Obsidian link formats:
    - Basic links: [[Note]]
    - Heading links: [[Note#Heading]]
    - Alias links: [[Real Name|Display Name]]
    """
    def __init__(self, vault_titles: List[str]):
        self.processor = KeywordProcessor(case_sensitive=False)
        self.titles_set = set(vault_titles)

        # Build dictionary: "Linux" -> "[[Linux]]"
        for title in vault_titles:
            # Skip very short generic names
            if len(title) < 3: continue

            clean_title = title.replace(".md", "")
            self.processor.add_keyword(clean_title, f"[[{clean_title}]]")

    def add_note(self, title: str):
        """Dynamically adds a new note title to the keyword processor."""
        if len(title) < 3: return
        clean_title = title.replace(".md", "")
        if clean_title not in self.titles_set:
            self.titles_set.add(clean_title)
            self.processor.add_keyword(clean_title, f"[[{clean_title}]]")

    @staticmethod
    def create_heading_link(note_title: str, heading: str) -> str:
        """Creates a link to a specific heading in a note.

        Args:
            note_title: The target note name
            heading: The heading within that note

        Returns:
            Formatted link: [[Note#Heading]]
        """
        return f"[[{note_title}#{heading}]]"

    @staticmethod
    def create_alias_link(real_name: str, alias: str) -> str:
        """Creates a link with a custom display name (alias).

        Args:
            real_name: The actual note name
            alias: The text to display instead

        Returns:
            Formatted link: [[Real Name|Alias]]
        """
        return f"[[{real_name}|{alias}]]"

    def process_text(self, text: str) -> str:
        """Injects wikilinks into text in a single pass.
        
        Logic:
        1. Protect existing links.
        2. Find all keyword matches.
        3. Replace ONLY the first occurrence of each unique keyword title (First Match Only).
        4. Restore original links.
        """
        import re

        # 1. Protect existing links
        link_pattern = r'\[\[([^\]]+)\]\]'
        existing_links = re.findall(link_pattern, text)
        placeholders = {}

        for i, match in enumerate(existing_links):
            placeholder = f"__LINK_PLACEHOLDER_{i}__"
            text = text.replace(f"[[{match}]]", placeholder, 1)
            placeholders[placeholder] = f"[[{match}]]"

        # 2. Find all matches (keyword, start, end)
        matches = self.processor.extract_keywords(text, span_info=True)
        
        # 3. Filter for first occurrence only
        # Matches are returned in order of appearance
        seen_titles = set()
        replacements = [] # (start, end, replacement_text)
        
        for title, start, end in matches:
            if title not in seen_titles:
                replacements.append((start, end, title))
                seen_titles.add(title)
        
        # 4. Apply replacements (reverse order to not mess up indices)
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        for start, end, link in replacements:
            text = text[:start] + link + text[end:]

        # 5. Restore original links
        for placeholder, original in placeholders.items():
            text = text.replace(placeholder, original)

        return text

class ObsidianGardener:
    """
    Manager for Vault operations: Auto-linking, Tagging, and Cleaning.
    Optimized for Cloud-Native branch: Lightweight, no local embeddings.
    """

    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = Path(vault_path) if vault_path else ProjectConfig.OBSIDIAN_VAULT
        self.logger = logging.getLogger("ObsidianGardener")
        
        # Initialize TagEngine and LinkOptimizer
        self.tag_engine = TagEngine()
        self.existing_notes = self._scan_vault()
        self.optimizer = LinkOptimizer(self.existing_notes)

    def generate_tags_for_content(self, content: str) -> List[str]:
        """Delegates tag generation to the TagEngine."""
        return self.tag_engine.generate_tags(content)

    def learn_note(self, filename: str):
        """Registers a new note with the LinkOptimizer to be seen immediately."""
        self.optimizer.add_note(filename)

    def refresh_optimizer(self):
        """Re-scans vault and rebuilds optimizer if file count changed."""
        current_titles = self._scan_vault()
        if len(current_titles) != len(self.existing_notes):
            self.existing_notes = current_titles
            self.optimizer = LinkOptimizer(self.existing_notes)
            self.logger.info(f"♻️ Refreshed LinkOptimizer index. Total notes: {len(self.existing_notes)}", extra={"tags": "GARDENER-REFRESH"})

    def _scan_vault(self) -> List[str]:
        """Index all note titles from the vault."""
        titles = []
        if not self.vault_path.exists():
            return titles
        for root, _, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith(".md"):
                    titles.append(file[:-3]) # Remove .md
        self.logger.info(f"Gardener indexed {len(titles)} notes for auto-linking.", extra={"tags": "GARDENER-INDEX"})
        return titles

    def update_dashboard(self):
        """
        Ensures 00_Dashboard.md exists and contains the Classic Minimalist layout.
        """
        try:
            dashboard_path = self.vault_path / "00_Dashboard.md"
            backup_path = self.vault_path / "00_Dashboard_Old.md"
            
            # --- Classic Dashboard Template ---
            today_str = datetime.datetime.now().strftime('%d %B %Y')
            intro_quote = "> _Minimalizm to nie brak czegoś, to po prostu odpowiednia ilość._"
            
            # Dataview Queries (Clean & Simple)
            inbox_query = create_dataview_query(
                query_type="TABLE",
                fields=[
                    'created as "Data"',
                    'choice(contains(tags, "#to-verify"), "🟠", choice(contains(file.tags, "#done"), "✅", choice(contains(file.tags, "#reading"), "📖", "⚪"))) as "Status"',
                    'tags as "Tagi"'
                ],
                from_source='"00_Skrzynka"',
                where='!contains(file.folder, "Archive") AND !contains(file.folder, "_BLEDY")',
                sort='created DESC',
                limit=10
            )

            tasks_query = """- [ ] #priority Priorytety na dziś
- [ ] #review Przejrzeć zaległe notatki"""

            content = f"""---
title: "Dashboard"
created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
type: dashboard
tags:
  - system
  - dashboard
---

<!-- AUTOGENERATED - Nie edytuj ręcznie, zmiany zostaną nadpisane -->

# Dashboard
_{today_str}_

{intro_quote}

---

## 📥 Skrzynka (Inbox)
{inbox_query}

## 🎯 Fokus
{tasks_query}

## 📚 Projekty (Ostatnio Modyfikowane)
```dataview
TABLE file.mtime as "Modyfikacja"
FROM !"00_Skrzynka" AND !"Dziennik"
SORT file.mtime DESC
LIMIT 5
```

---
"""
            
            # Check logic
            if dashboard_path.exists():
                current_text = dashboard_path.read_text(encoding='utf-8')
                if "<!-- AUTOGENERATED" not in current_text:
                    self.logger.info("Backing up legacy dashboard...")
                    shutil.move(str(dashboard_path), str(backup_path))
            
            # Write new dashboard
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            self.logger.info("Updated 00_Dashboard.md (Classic Style).", extra={"tags": "GARDENER-DASHBOARD"})

        except Exception as e:
            self.logger.error(f"Dashboard update failed: {e}")

    def update_daily_log(self, title: str, summary: str, tasks: List[str], note_path: str = None):
        """
        Appends a processing report to the Daily Note with proper formatting.
        """
        today = datetime.date.today().strftime("%Y-%m-%d")
        daily_folder = self.vault_path / "Daily"
        daily_folder.mkdir(parents=True, exist_ok=True)
        daily_path = daily_folder / f"{today}.md"

        # Ensure Daily Note exists with proper template
        if not daily_path.exists():
            daily_builder = DailyNoteBuilder(datetime.datetime.now())
            daily_path.write_text(daily_builder.build(), encoding='utf-8')

        # Create Log Entry with proper formatting
        timestamp = datetime.datetime.now().strftime("%H:%M")
        link = f"[[{title}]]" if title else "Nieznana notatka"

        # Truncate summary properly
        clean_summary = summary[:300]
        if len(summary) > 300:
            clean_summary = clean_summary.rsplit(' ', 1)[0] + '...'

        log_entry = f"\n---\n\n### 🤖 {timestamp} — Przetworzono: {link}\n\n"
        log_entry += f"> [!note] Podsumowanie\n> {clean_summary}\n\n"

        if tasks:
            trainings = [t for t in tasks if "#Szkolenia" in t or "#szkolenia" in t]
            others = [t for t in tasks if "#Szkolenia" not in t and "#szkolenia" not in t]

            if others:
                log_entry += "#### 🛠️ Wykryte zadania\n\n"
                for task in others:
                    log_entry += f"- [ ] {task}\n"
                log_entry += "\n"

            if trainings:
                log_entry += "#### 📚 Materiały ze szkolenia\n\n"
                for task in trainings:
                    log_entry += f"- [ ] {task}\n"
                log_entry += "\n"
        else:
            log_entry += "_Brak wykrytych zadań._\n\n"

        # Append to file
        try:
            with open(daily_path, "a", encoding="utf-8") as f:
                f.write(log_entry)

            self.logger.info(f"Updated Daily Note: {daily_path}", extra={"tags": "GARDENER-DAILY"})

        except Exception as e:
            self.logger.error(f"Failed to update Daily Note: {e}")

    def archive_source_file(self, source_path: str, subfolder: str = "Audio"):
        """Archives the source file to Resources folder."""
        try:
            src = Path(source_path)
            if not src.exists():
                return
                
            archive_dir = self.vault_path / "Zasoby" / subfolder
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            dest = archive_dir / src.name
            shutil.move(str(src), str(dest))
            self.logger.info(f"Archived file to: {dest}", extra={"tags": "GARDENER-ARCHIVE"})
        except Exception as e:
            self.logger.error(f"Failed to archive file: {e}")

    def process_file(self, file_path: str) -> Tuple[bool, str]:
        """Reads, links, tags and saves a specific note."""
        try:
            path = Path(file_path)
            if not path.exists(): return False, "File not found."
            
            content = path.read_text(encoding='utf-8')
            original_content = content
            
            # 1. FlashText Auto-linking (Fast) - Literal matching only
            new_content = self.optimizer.process_text(content)
            
            # 2. Smart Tagging
            # Check if we have tags in frontmatter
            has_tags = False
            import yaml
            if new_content.startswith("---"):
                try:
                    end_idx = new_content.find("\n---", 3)
                    if end_idx != -1:
                        frontmatter_str = new_content[3:end_idx]
                        meta = yaml.safe_load(frontmatter_str)
                        if meta and "tags" in meta and meta["tags"]:
                            has_tags = True
                except Exception: pass
            
            if not has_tags:
                generated_tags = self.tag_engine.generate_tags(new_content)
                if generated_tags:
                    self.logger.info(f"Generated tags for {path.name}: {generated_tags}")
                    if new_content.startswith("---"):
                        # Update existing frontmatter
                        end_idx = new_content.find("\n---", 3)
                        frontmatter_str = new_content[3:end_idx]
                        try:
                            meta = yaml.safe_load(frontmatter_str) or {}
                            meta["tags"] = generated_tags
                            new_frontmatter = yaml.dump(meta, allow_unicode=True).strip()
                            new_content = "---\n" + new_frontmatter + "\n" + new_content[end_idx+1:]
                        except Exception: pass
                    else:
                        # Add new frontmatter
                        frontmatter = "---\ntags:\n"
                        for t in generated_tags:
                            frontmatter += f"  - {t}\n"
                        frontmatter += "---\n\n"
                        new_content = frontmatter + new_content

            if new_content != original_content:
                path.write_text(new_content, encoding='utf-8')
                return True, "Auto-linking and Tagging applied."
            
            return True, "No changes needed."
        except Exception as e:
            self.logger.error(f"Gardener failed for {file_path}: {e}")
            return False, str(e)

    # --- Compatibility Methods for app.py ---

    def auto_link(self, text: str) -> str:
        """Wrapper for FlashText optimizer to support app.py."""
        return self.optimizer.process_text(text)

    def smart_tagging(self, tags: List[str], content: str = "") -> List[str]:
        """
        Deduplicates, normalizes, and enriches tags using TagEngine.
        """
        # 1. Start with existing tags
        all_tags = set()
        for t in tags:
            t = t.strip().lower().lstrip('#')
            if t:
                all_tags.add(t)
        
        # 2. Add tags from TagEngine if content is provided
        if content:
            engine_tags = self.tag_engine.generate_tags(content)
            all_tags.update(engine_tags)
            
        return sorted(list(all_tags))

    def save_note(self, title: str, content: str, tags: list, summary: str) -> Path:
        """Saves a markdown note to the vault with properly formatted YAML frontmatter."""
        # Sanitize filename
        safe_filename = sanitize_filename(title)
        full_path = self.vault_path / f"{safe_filename}.md"

        # Use TranscriptNoteBuilder for transcript-specific formatting
        builder = TranscriptNoteBuilder(title=title, tags=tags, source_type="audio") # Assuming audio for now
        builder.add_summary(summary)
        builder.add_transcript(content)

        final_content = builder.build()

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(final_content)

        logger.info(f"Note saved with YAML: {full_path}", extra={"tags": "OBSIDIAN-SAVE"})

        # Update Dashboard & MOCs
        self.update_dashboard()
        self.generate_index()

        return full_path

    def smart_categorize(self, content: str) -> str:
        """
        Uses LLM to decide which folder the note belongs to.
        """
        try:
            import ollama
            from config import ProjectConfig
            
            # Use configurable categories
            categories = getattr(ProjectConfig, "CATEGORIES", ["Edukacja", "Newsy", "Badania", "Zasoby", "Dziennik", "Prywatne", "Przegląd"])
            
            prompt = f"""
            Przeanalizuj treść notatki i wybierz JEDNĄ najbardziej pasującą kategorię z listy: {', '.join(categories)}.
            Jeśli nie masz pewności lub treść wymaga weryfikacji, wybierz 'Przegląd'.
            Zwróć tylko nazwę kategorii, nic więcej.
            
            Treść:
            {content[:2000]}
            """
            
            response = ollama.chat(
                model=ProjectConfig.OLLAMA_MODEL_FAST,
                messages=[{'role': 'user', 'content': prompt}]
            )
            choice = response['message']['content'].strip()
            
            # Clean up response (sometimes LLM adds quotes or dots)
            for cat in categories:
                if cat.lower() in choice.lower():
                    return cat
            return "Zasoby" # Default if unsure
        except Exception as e:
            self.logger.error(f"Categorization failed: {e}")
            return "Zasoby"

    def generate_index(self):
        """
        Generates categorized Maps of Content (MOCs) based on tags hierarchy in tags.yaml.
        Traverses ProjectConfig.TAGS_CONFIG to map sub-tags to parent pillars.
        """
        import yaml
        import re
        
        try:
            if not ProjectConfig.TAGS_CONFIG:
                self.logger.warning("MOC Generation skipped: No tags configuration found.")
                return

            # 1. Build Tag -> Pillar Mapping
            # Example: "dora" -> "compliance", "osint" -> "cybersec"
            tag_to_pillar = {}
            pillar_meta = {} # pillar_key -> {filename, title}

            def map_node(node_key, node_data, pillar_key):
                # Map the key itself (e.g. "dora" -> "compliance")
                tag_to_pillar[node_key.lower()] = pillar_key
                
                # Process current node's tags
                for tag in node_data.get("tags", []):
                    clean_tag = tag.lstrip('#').lower()
                    tag_to_pillar[clean_tag] = pillar_key
                
                # Recurse subcategories
                subcats = node_data.get("subcategories", {})
                for sub_key, sub_data in subcats.items():
                    map_node(sub_key, sub_data, pillar_key)

            for p_key, p_data in ProjectConfig.TAGS_CONFIG.items():
                # Extract pillar metadata
                filename = f"000_MOC_{p_key.capitalize()}.md"
                title = f"🏷️ {p_key.capitalize()} MOC"
                
                # If the pillar has subcategories or specific keywords, we use it
                # We can also look for custom titles if we add them to YAML later
                pillar_meta[p_key] = {"filename": filename, "title": title}
                map_node(p_key, p_data, p_key)

            # 2. Bucket for notes: pillar -> {tag: [links]}
            moc_data = {p_key: {} for p_key in pillar_meta}
            
            # 3. Scan Vault
            for root, _, files in os.walk(self.vault_path):
                for file in files:
                    if not file.endswith(".md") or file.startswith("000_MOC"): continue
                    
                    path = Path(root) / file
                    try:
                        content = path.read_text(encoding='utf-8')
                        file_tags = set()

                        # A. Extract YAML Tags
                        if content.startswith("---"):
                            try:
                                end_idx = content.find("\n---", 3)
                                if end_idx != -1:
                                    frontmatter_str = content[3:end_idx]
                                    meta = yaml.safe_load(frontmatter_str)
                                    if meta and "tags" in meta:
                                        tags_val = meta["tags"]
                                        if isinstance(tags_val, list):
                                            for t in tags_val:
                                                if isinstance(t, str): file_tags.add(t.lower())
                                        elif isinstance(tags_val, str):
                                            file_tags.add(tags_val.lower())
                            except Exception: pass

                        # B. Extract Inline Hashtags
                        inline_tags = re.findall(r'#([\w/-]+)', content)
                        for t in inline_tags:
                            file_tags.add(t.lower())
                        
                        link = f"[[{file.replace('.md', '')}]]"
                        
                        # Assign note to MOCs based on mapping
                        for tag in file_tags:
                            tag_clean = tag.lstrip('#')
                            
                            # Find which pillar this tag belongs to
                            # Matches "compliance/dora" -> "compliance" if "compliance" is a key
                            # or "dora" -> "compliance" if explicitly mapped
                            target_pillar = None
                            if tag_clean in tag_to_pillar:
                                target_pillar = tag_to_pillar[tag_clean]
                            else:
                                # Fallback: check if any pillar key is a prefix
                                for p_key in pillar_meta:
                                    if tag_clean.startswith(p_key):
                                        target_pillar = p_key
                                        break
                            
                            if target_pillar:
                                display_tag = f"#{tag_clean}"
                                if display_tag not in moc_data[target_pillar]:
                                    moc_data[target_pillar][display_tag] = []
                                moc_data[target_pillar][display_tag].append(link)

                    except Exception as e:
                        self.logger.warning(f"Error processing file {file}: {e}")

            # 4. Generate MOC Files
            for p_key, data in moc_data.items():
                if not data: continue
                
                meta = pillar_meta[p_key]
                file_path = self.vault_path / meta["filename"]
                
                md = f"# {meta['title']}\n> Auto-generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                
                for subtag, links in sorted(data.items()):
                    unique_links = sorted(list(set(links)))
                    md += f"## {subtag} ({len(unique_links)})\n"
                    for link in unique_links:
                        md += f"- {link}\n"
                    md += "\n"
                
                file_path.write_text(md, encoding='utf-8')
                
            self.logger.info("Regenerated Hierarchical MOCs.")

        except Exception as e:
            self.logger.error(f"MOC Generation failed: {e}")

