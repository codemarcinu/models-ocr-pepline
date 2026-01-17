import feedparser
import os
import json
import ollama
import logging
import asyncio
import re
import requests
import edge_tts
from datetime import datetime
from typing import Set, Optional, List, Dict, Any
from pathlib import Path

from config import ProjectConfig, logger
from ai_research import WebResearcher
from adapters.obsidian.vault_manager import ObsidianGardener
from core.services.rag_service import ObsidianRAG
from adapters.google.gemini_adapter import GeminiBrain

class NewsAgent:
    """
    Advanced Cybersec Analyst Agent (V2).
    Features:
    - Semantic Filtering (Value over Volume)
    - CVE Enrichment (NIST/CIRCL)
    - RAG Context Check ("Does this affect me?")
    - Daily Digest Format
    - Audio Briefing Generation
    """
    
    RSS_FEEDS = {
        "Sekurak": "https://feeds.feedburner.com/sekurak",
        "Niebezpiecznik": "https://feeds.feedburner.com/niebezpiecznik",
        "Zaufana Trzecia Strona": "https://zaufanatrzeciastrona.pl/feed/",
        "The Hacker News": "https://feeds.feedburner.com/TheHackersNews"
    }

    def __init__(self):
        self.news_dir = ProjectConfig.OBSIDIAN_VAULT / "Newsy"
        self.news_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = ProjectConfig.BASE_DIR / "processed_news.json"
        
        # Models
        self.model = ProjectConfig.OLLAMA_MODEL # Heavy (Summarization - Fallback)
        self.fast_model = ProjectConfig.OLLAMA_MODEL_FAST # Light (Filtering)
        
        self.gardener = ObsidianGardener()
        self.researcher = WebResearcher(gardener=self.gardener)
        self.rag = ObsidianRAG() # For cross-checking impact
        self.gemini = GeminiBrain() # Cloud Acceleration

    def _load_history(self) -> Set[str]:
        if self.history_file.exists():
            try:
                return set(json.loads(self.history_file.read_text()))
            except:
                return set()
        return set()

    def _save_history(self, history: Set[str]):
        self.history_file.write_text(json.dumps(list(history)))

    # --- 1. Semantic Filter ---
    def _is_relevant(self, title: str, summary_snippet: str) -> bool:
        """
        Uses a lightweight model to filter out non-technical fluff.
        """
        prompt = ProjectConfig.PROMPTS.get('news_agent', {}).get('is_relevant_prompt', "")
        try:
            resp = ollama.chat(
                model=self.fast_model,
                messages=[{'role': 'user', 'content': f"{prompt}\n\nTitle: {title}\nSnippet: {summary_snippet}"}]
            )
            decision = resp['message']['content'].strip().upper()
            return "YES" in decision
        except Exception as e:
            logger.warning(f"Filter error: {e}")
            return True # Fail open (keep it just in case)

    # --- 2. CVE Enrichment ---
    def _extract_and_enrich_cves(self, text: str) -> List[Dict]:
        """
        Finds CVE-YYYY-NNNN patterns and fetches CVSS scores.
        """
        cve_pattern = r'CVE-\d{4}-\d+'
        cves = list(set(re.findall(cve_pattern, text)))
        enriched_data = []

        for cve in cves[:3]: # Limit API calls
            try:
                # Using CIRCL.lu API (No key required, polite limits)
                response = requests.get(f"https://cve.circl.lu/api/cve/{cve}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    cvss = data.get('cvss', 'N/A')
                    summary = data.get('summary', 'No summary')
                    enriched_data.append({
                        "id": cve,
                        "cvss": cvss,
                        "summary": summary
                    })
            except Exception as e:
                logger.warning(f"CVE lookup failed for {cve}: {e}")
        
        return enriched_data

    # --- 3. RAG Context Check ---
    def _check_impact(self, text: str) -> str:
        """
        Queries the Vault to see if the user uses technologies mentioned in the news.
        """
        try:
            # Extract keywords first
            prompt_template = ProjectConfig.PROMPTS.get('news_agent', {}).get('check_impact_keywords_prompt', "")
            keywords_resp = ollama.chat(
                model=self.fast_model,
                messages=[{'role': 'user', 'content': prompt_template.format(text=text[:1000])}]
            )
            keywords = keywords_resp['message']['content'].strip()
            
            # Query RAG
            query = f"Czy używam technologii: {keywords}? Czy mam projekty z tym związane?"
            rag_response = self.rag.query(query, n_results=3)
            
            if "Brak odpowiednich notatek" in rag_response:
                return ""
            
            return f"\n> ⚠️ **Analiza Wpływu (RAG):**\n> System wykrył potencjalne powiązania w Twojej bazie wiedzy:\n> {rag_response[:300]}...\n"
        except Exception:
            return ""

    # --- 4. Core Analysis ---
    def _process_article(self, entry) -> Optional[Dict]:
        # 1. Fetch
        _, content = self.researcher.fetch_article_content(entry.link)
        if not content: return None

        # 2. Summarize (Hybrid: Cloud Gemini > Local Ollama)
        summary = None
        
        # Try Gemini First
        if self.gemini.available:
            logger.info(f"Summarizing via Gemini: {entry.title}")
            summary = self.gemini.summarize_article(content)
        
        # Fallback to Ollama
        if not summary:
            logger.info(f"Summarizing via Ollama (Fallback): {entry.title}")
            prompt = ProjectConfig.PROMPTS.get('news_agent', {}).get('process_article_system_prompt', "")
            try:
                resp = ollama.chat(model=self.model, messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': f"Tytuł: {entry.title}\n\nTreść: {content[:8000]}"}
                ])
                summary = resp['message']['content']
            except Exception as e:
                logger.error(f"Local summarization failed: {e}")
                return None

        # 3. Enrich
        cves = self._extract_and_enrich_cves(content)
        impact_note = self._check_impact(summary) # Check summary is faster/cleaner
        
        return {
            "title": entry.title,
            "url": entry.link,
            "summary": summary,
            "cves": cves,
            "impact": impact_note,
            "source": entry.get('source', {}).get('title', 'RSS')
        }

    # --- 5. Output Generation ---
    async def _generate_audio_briefing(self, digest_text: str, date_str: str):
        """Generates an MP3 podcast from the digest."""
        try:
            # Clean text for TTS (remove heavy markdown)
            clean_text = digest_text.replace("#", "").replace("*", "").replace(">", "")
            clean_text = f"Cześć Marcin. Oto Twój raport bezpieczeństwa na dzień {date_str}. " + clean_text
            
            voice = "pl-PL-MarekNeural"
            output_path = ProjectConfig.OBSIDIAN_VAULT / "00_Inbox" / f"Briefing-{date_str}.mp3"
            
            communicate = edge_tts.Communicate(clean_text[:4000], voice) # Limit chars for safety
            await communicate.save(str(output_path))
            logger.info(f"Audio briefing generated: {output_path}")
        except Exception as e:
            logger.error(f"TTS Generation failed: {e}")

    def _save_daily_digest(self, articles: List[Dict]):
        if not articles: return

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date_str}-Cyber-Briefing.md"
        path = self.news_dir / filename
        
        # Header
        content = f"# 🛡️ Cyber Briefing: {date_str}\n**Zebrane Newsy:** {len(articles)} | **Status:** Wygenerowano automatycznie\n\n---\n"
        tts_buffer = ""

        # Body
        for article in articles:
            # CVE Badge
            cve_section = ""
            if article['cves']:
                for cve in article['cves']:
                    score = float(cve['cvss']) if cve['cvss'] != 'N/A' else 0
                    color = "🔴" if score >= 9.0 else "🟡" if score >= 7.0 else "🟢"
                    cve_section += f"{color} **{cve['id']}** (CVSS: {cve['cvss']}) "
                cve_section += "\n"

            content += f"\n## {article['title']}\n"
            content += f"🔗 [Link do źródła]({article['url']}) | 📰 {article['source']}\n\n"
            content += cve_section
            content += f"{article['summary']}\n"
            content += f"{article['impact']}\n"
            content += "---\n"

            # Buffer for Audio (shorter version)
            tts_buffer += f"News: {article['title']}. {article['summary'][:200]}. "

        # Save MD
        if path.exists():
            # Append if exists (in case ran multiple times a day)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content.replace(f"# 🛡️ Cyber Briefing: {date_str}", "")) # Remove header for append
        else:
            path.write_text(content, encoding='utf-8')
        
        # Link
        self.gardener.process_file(str(path))
        logger.info(f"Daily Digest saved: {path}")

        # Generate Audio (Async wrapper)
        asyncio.run(self._generate_audio_briefing(tts_buffer, date_str))

    def run(self, limit: int = 5):
        """Main Orchestrator."""
        history = self._load_history()
        articles_buffer = []
        
        logger.info("NewsAgent started. Scanning feeds...", extra={"tags": "NEWS-START"})
        
        for source, url in self.RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                logger.info(f"Feed: {source} - Found {len(feed.entries)} entries.")
                
                processed_count = 0
                for entry in feed.entries:
                    if processed_count >= limit: break
                    if entry.link in history: continue
                    
                    # 1. Semantic Filter
                    if not self._is_relevant(entry.title, entry.get('summary', '')):
                        logger.info(f"Skipped (Irrelevant): {entry.title}")
                        history.add(entry.link) # Mark as seen to skip next time
                        continue
                        
                    logger.info(f"Processing: {entry.title}")
                    
                    # 2. Process (Scrape, Summarize, Enrich, Impact)
                    article_data = self._process_article(entry)
                    
                    if article_data:
                        articles_buffer.append(article_data)
                        history.add(entry.link)
                        processed_count += 1
                        
            except Exception as e:
                logger.error(f"Feed error ({source}): {e}")
        
        # 3. Finalize
        if articles_buffer:
            self._save_daily_digest(articles_buffer)
            self._save_history(history)
            logger.info(f"Job Done. Processed {len(articles_buffer)} articles.")
        else:
            logger.info("No new relevant articles found.")

        return len(articles_buffer)

if __name__ == "__main__":
    agent = NewsAgent()
    agent.run(limit=3)