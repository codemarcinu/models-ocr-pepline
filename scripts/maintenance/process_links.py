import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Set

from config import ProjectConfig, logger
from adapters.google.gemini_adapter import GeminiBrain
from ai_research import WebResearcher

class DailyLinkProcessor:
    """
    Scans the Daily Note for links and processes them using Gemini 1.5 Flash.
    """
    
    def __init__(self):
        self.vault_path = ProjectConfig.OBSIDIAN_VAULT
        self.daily_dir = self.vault_path / "Daily"
        self.inbox_dir = ProjectConfig.INBOX_DIR
        self.history_file = ProjectConfig.BASE_DIR / "processed_links_history.json"
        
        self.gemini = GeminiBrain()
        self.researcher = WebResearcher() # Used for fetching content
        self.processed_history = self._load_history()

    def _load_history(self) -> Set[str]:
        if self.history_file.exists():
            try:
                return set(json.loads(self.history_file.read_text()))
            except:
                return set()
        return set()

    def _save_history(self):
        self.history_file.write_text(json.dumps(list(self.processed_history)))

    def get_todays_note(self) -> Path:
        today_str = datetime.now().strftime("%Y-%m-%d")
        return self.daily_dir / f"{today_str}.md"

    def extract_links(self, note_path: Path) -> List[str]:
        """
        Extracts links from the '## Prasówka' or '## Gemini Queue' section.
        """
        if not note_path.exists():
            logger.warning(f"Daily note not found: {note_path}")
            return []

        content = note_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        links = []
        in_section = False
        
        # simple state machine to find section
        for line in lines:
            if line.strip().startswith("## "):
                header = line.strip().lower()
                if "prasówka" in header or "gemini queue" in header:
                    in_section = True
                else:
                    in_section = False
                continue
            
            if in_section:
                # Regex to find http/https links
                found = re.findall(r'(https?://[^\s\)]+)', line)
                for link in found:
                    links.append(link)
        
        return links

    def save_summary(self, article_data: dict):
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = self.inbox_dir / f"PRASOWKA_{date_str}.md"
        
        mode = 'a' if output_file.exists() else 'w'
        
        with open(output_file, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(f"# 🗞️ Prasówka Gemini: {date_str}\n\n")
            
            f.write(f"## {article_data['title']}\n")
            f.write(f"🔗 {article_data['url']}\n\n")
            f.write(f"{article_data['summary']}\n")
            f.write(f"\n---\n")
            
        logger.info(f"Saved summary for {article_data['title']}")

    def run(self):
        if not self.gemini.available:
            logger.error("Gemini not available. Aborting.")
            return

        note_path = self.get_todays_note()
        logger.info(f"Scanning: {note_path}")
        
        links = self.extract_links(note_path)
        new_links = [l for l in links if l not in self.processed_history]
        
        if not new_links:
            logger.info("No new links to process.")
            return

        logger.info(f"Found {len(new_links)} new links.")
        
        for url in new_links:
            try:
                # 1. Fetch
                title, content = self.researcher.fetch_article_content(url)
                if not content:
                    logger.info(f"Skipping inaccessible link: {url}")
                    self.processed_history.add(url) # Skip next time
                    continue
                
                # 2. Summarize (Gemini)
                summary = self.gemini.summarize_article(content)
                if not summary:
                    logger.warning(f"Failed to summarize {url}")
                    continue
                
                # 3. Save
                self.save_summary({
                    "title": title,
                    "url": url,
                    "summary": summary
                })
                
                # 4. Update History
                self.processed_history.add(url)
                self._save_history()
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")

if __name__ == "__main__":
    processor = DailyLinkProcessor()
    processor.run()
