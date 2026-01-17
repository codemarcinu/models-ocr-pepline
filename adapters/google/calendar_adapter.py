import os
import datetime
import logging
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import httplib2
httplib2.debuglevel = 4

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import ollama

from config import ProjectConfig

# Setup logger
logger = logging.getLogger("CalendarBridge")

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/tasks'
]

class CalendarBridge:
    def __init__(self, vault_path: str = None):
        self.vault_path = Path(vault_path) if vault_path else ProjectConfig.OBSIDIAN_VAULT
        self.creds = None
        self.service = None
        self.tasks_service = None
        self.model = ProjectConfig.OLLAMA_MODEL_FAST  # Use fast model for extraction
        self.credentials_path = ProjectConfig.BASE_DIR / "credentials.json"
        self.token_path = ProjectConfig.BASE_DIR / "token.json"

    def authenticate(self):
        """Shows basic usage of the Google Calendar API."""
        if self.token_path.exists():
            self.creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Error refreshing token: {e}")
                    self.creds = None
            
            if not self.creds:
                if not self.credentials_path.exists():
                    logger.error(f"Credentials file not found at {self.credentials_path}. Please download it from Google Cloud Console.")
                    return False
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), SCOPES)
                    self.creds = flow.run_local_server(port=0)
                except Exception as e:
                     logger.error(f"OAuth flow failed: {e}")
                     return False

            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(self.creds.to_json())

        try:
            self.service = build('calendar', 'v3', credentials=self.creds)
            self.tasks_service = build('tasks', 'v1', credentials=self.creds)
            return True
        except Exception as e:
            logger.error(f"Failed to build service: {e}")
            return False

    def fetch_pending_tasks(self) -> List[str]:
        """Pobiera nieukończone zadania z domyślnej listy Google Tasks."""
        if not self.tasks_service:
            if not self.authenticate():
                logger.error("Authentication failed. Cannot fetch tasks.")
                return []
        
        try:
            results = self.tasks_service.tasks().list(tasklist='@default', showCompleted=False).execute()
            items = results.get('items', [])

            if not items:
                logger.info("Brak nowych zadań w Google Tasks.")
                return []

            tasks_md = []
            for item in items:
                title = item['title']
                notes = item.get('notes', '')
                
                # Formatuj jako Markdown checkbox
                task_line = f"- [ ] {title} #google_tasks"
                if notes:
                    # Escape newlines in notes for single line display or handle differently
                    clean_notes = notes.replace('\n', ' ')
                    task_line += f" 📝 *{clean_notes}*"
                
                tasks_md.append(task_line)
            
            return tasks_md

        except Exception as e:
            logger.error(f"Błąd podczas pobierania Google Tasks: {e}")
            return []

    def scan_for_events(self):
        """
        Scans all markdown files in the vault for the #calendar tag 
        and specifically parses lines in '## 📅 Calendar' section.
        """
        if not self.service:
            if not self.authenticate():
                logger.error("Authentication failed. Cannot scan.")
                return

        logger.info(f"Scanning vault: {self.vault_path}")
        
        # Simple recursion
        for md_file in self.vault_path.rglob("*.md"):
            self._process_file(md_file)

    def _process_file(self, file_path: Path):
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # Quick check if file needs processing
        if "#calendar" not in content and "## 📅 Calendar" not in content:
            return

        # Look for unsynced events
        # We assume the AI generates lines like "SPOTKANIE: Dentist..." or "TERMIN: Deadline..."
        # We want to match lines that DO NOT already have a (Synced) marker or link.
        
        lines = content.split('\n')
        new_content_lines = []
        modified = False
        in_calendar_section = False

        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith("## 📅 Calendar"):
                in_calendar_section = True
                new_content_lines.append(line)
                continue
            elif stripped.startswith("##"):
                in_calendar_section = False
                new_content_lines.append(line)
                continue

            # Process only if we are in the calendar section
            is_event_line = False
            
            if in_calendar_section:
                upper_line = stripped.upper()
                # 1. Check for explicit prefixes (Case Insensitive)
                if upper_line.startswith("SPOTKANIE") or upper_line.startswith("TERMIN") or upper_line.startswith("WIZYTA"):
                    is_event_line = True
                # 2. Check for implicit events (starts with dash + contains time-like info)
                elif (stripped.startswith("-") or stripped.startswith("*")) and len(stripped) > 10:
                     # Check for time indicators (e.g. 17:00, 17.00, jutro, dzisiaj)
                     if any(x in upper_line for x in [":", "JUTRO", "DZISIAJ", "PONIEDZIAŁEK", "WTOREK", "ŚRODA", "CZWARTEK", "PIĄTEK", "SOBOTA", "NIEDZIELA"]):
                         is_event_line = True
            
            # Also support lines tagged explicitly anywhere
            if "#event" in stripped or "#kalendarz" in stripped:
                is_event_line = True

            if is_event_line and "[Synced]" not in line and "✅" not in line:
                logger.info(f"Found unsynced event in {file_path.name}: {stripped}")
                event_data = self._parse_event_with_llm(stripped, file_path.name)
                
                if event_data:
                    link = self._add_to_google_calendar(event_data)
                    if link:
                        # Append the link and synced status to the line
                        line = f"{line} [Synced]({link})"
                        modified = True
                
            new_content_lines.append(line)

        if modified:
            # Check if we should remove the #calendar tag from frontmatter or body if strictly needed?
            # For now, we leave the tag but mark items as synced.
            # Optionally, we could replace #calendar with #calendar/synced in the file.
            final_content = "\n".join(new_content_lines)
            final_content = final_content.replace("#calendar", "#calendar/synced")
            file_path.write_text(final_content, encoding='utf-8')
            logger.info(f"Updated file: {file_path}")

    def _parse_event_with_llm(self, text: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Uses Ollama to extract JSON event data.
        """
        prompt = f"""
        Extract event details from the text below into a valid JSON object.
        Current Year: {datetime.datetime.now().year}
        Today: {datetime.datetime.now().strftime("%Y-%m-%d")}
        
        TEXT: "{text}"
        CONTEXT FILE: "{context}"

        OUTPUT JSON FORMAT:
        {{
            "summary": "Short title",
            "start": "ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS)",
            "end": "ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS) - default to start + 1 hour if not specified",
            "location": "optional string"
        }}
        
        Reply ONLY with JSON.
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            clean_json = response['message']['content'].strip()
            # Try to find JSON block if wrapped in markdown
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
                
            return json.loads(clean_json)
        except Exception as e:
            logger.error(f"LLM Parsing failed for '{text}': {e}")
            return None

    def _add_to_google_calendar(self, event_data: Dict[str, Any]) -> Optional[str]:
        if not self.service: return None
        
        event = {
            'summary': event_data.get('summary', 'New Event'),
            'location': event_data.get('location', ''),
            'description': 'Created from Obsidian AI System',
            'start': {
                'dateTime': event_data.get('start'),
                'timeZone': 'Europe/Warsaw', # Default, could be configurable
            },
            'end': {
                'dateTime': event_data.get('end'),
                'timeZone': 'Europe/Warsaw',
            },
        }

        try:
            event_result = self.service.events().insert(calendarId='primary', body=event).execute()
            logger.info(f"Event created: {event_result.get('htmlLink')}")
            return event_result.get('htmlLink')
        except Exception as e:
            logger.error(f"Google API Error: {e}")
            return None

    def run_forever(self, interval_seconds: int = 600):
        """Runs the scan periodically."""
        logger.info(f"Starting CalendarBridge service (interval: {interval_seconds}s)")
        while True:
            try:
                self.scan_for_events()
            except Exception as e:
                logger.error(f"Error during scheduled scan: {e}")
            
            import time
            time.sleep(interval_seconds)

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    bridge = CalendarBridge()
    
    if "--service" in sys.argv:
        bridge.run_forever()
    else:
        bridge.scan_for_events()
