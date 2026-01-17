import os
import re
import ollama
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm
from pathlib import Path

from config import ProjectConfig, logger
from utils.tag_engine import TagEngine
from adapters.google.gemini_adapter import UniversalBrain

class TranscriptProcessor:
    """
    Refactored Note Generator: Converts raw transcripts into structured technical documentation.
    Optimized for Cloud-Native: Uses UniversalBrain (OpenAI/Gemini) for intelligence.
    """
    
    def __init__(self):
        self.brain = UniversalBrain()
        self.logger = logging.getLogger("TranscriptProcessor")
        self.prompts = ProjectConfig.PROMPTS.get('ai_notes', {})
        self.tag_engine = TagEngine()

    def _generate_metadata(self, text: str) -> Tuple[str, str]:
        """Uses Cloud AI to generate title and short summary."""
        if not self.brain.available:
            return "Note-" + datetime.now().strftime("%Y%m%d-%H%M"), "Automatyczna notatka."

        prompt = self.prompts.get('metadata_prompt', "Na podstawie tekstu podaj: 1. Krótki tytuł techniczny, 2. Jednozdaniowe podsumowanie.")
        system_prompt = "Jesteś asystentem technicznym. Wygeneruj metadane dla notatki."
        user_prompt = f"{prompt}\n\nTekst: {text[:4000]}"
        try:
            content = self.brain.generate_content(user_prompt, system_prompt=system_prompt)
            if not content: return "Note-" + datetime.now().strftime("%Y%m%d-%H%M"), "Brak odpowiedzi AI."
            
            lines = content.split('\n')
            title = lines[0].strip().replace("1. ", "").replace("Tytuł: ", "")
            summary = lines[1].strip().replace("2. ", "").replace("Podsumowanie: ", "") if len(lines) > 1 else "Brak podsumowania."
            # Sanitize title
            title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
            return title, summary
        except Exception as e:
            self.logger.error(f"Metadata generation failed: {e}")
            return "Note-" + datetime.now().strftime("%Y%m%d-%H%M"), "Automatyczna notatka."

    def generate_note_content_from_text(self, text: str, meta: Dict[str, Any] = None, style: str = "Academic") -> Dict[str, Any]:
        """
        Direct generation from text string using Cloud AI.
        """
        if not text:
            return {"title": "Empty Note", "content": "", "tags": []}

        # 1. Metadata
        title, summary = self._generate_metadata(text)
        if meta and meta.get('title') and meta.get('title') != "Unknown Title":
             # Prefer metadata title but sanitize it
             title = "".join(c for c in meta['title'] if c.isalnum() or c in " -_").strip()

        # 2. Context Chunking (larger chunks for Cloud AI)
        chunks = [text[i:i+15000] for i in range(0, len(text), 14000)]
        full_body = []

        # Adjust system prompt based on style
        style_instruction = ""
        if style == "Bullet Points": style_instruction = "Używaj głównie list wypunktowanych."
        if style == "Summary": style_instruction = "Skup się tylko na najważniejszych wnioskach."
        
        system_prompt = self.prompts.get('system_prompt', "") + f"\nSTYL: {style_instruction}"

        for i, chunk in enumerate(tqdm(chunks, desc="Cloud Refining Content")):
            try:
                user_prompt = f"Przetwórz fragment {i+1} (zachowaj ciągłość):\n{chunk}"
                res = self.brain.generate_content(user_prompt, system_prompt=system_prompt)
                if res:
                    full_body.append(res)
            except Exception as e:
                self.logger.error(f"Chunk error: {e}")

        combined_body = "\n\n".join(full_body)
        
        # 3. Hybrid Tagging
        tags = self.tag_engine.generate_tags(combined_body)
        
        if meta:
            source_tag = f"source/{meta.get('uploader', 'unknown').lower().replace(' ', '_')}"
            if source_tag not in tags:
                tags.append(source_tag)

        return {
            "title": title,
            "content": combined_body,
            "summary": summary,
            "tags": tags
        }

    # Legacy wrapper for compatibility if needed, but App uses the method above now
    def generate_note_content(self, transcript_file: str) -> Dict[str, Any]:
        path = Path(transcript_file)
        if not path.exists(): return {"error": "File not found"}
        return self.generate_note_content_from_text(path.read_text(encoding='utf-8'))
