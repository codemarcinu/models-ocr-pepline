import logging
import os.path
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from config import ProjectConfig

logger = logging.getLogger("KeepApiAdapter")

SCOPES = ['https://www.googleapis.com/auth/keep.readonly']

class KeepApiAdapter:
    def __init__(self):
        self.creds = None
        self.service = None
        self.credentials_path = ProjectConfig.BASE_DIR / "credentials.json"
        self.token_path = ProjectConfig.TOKEN_KEEP_API_FILE
        self.inbox_dir = ProjectConfig.INBOX_DIR / "Keep"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def authenticate(self):
        """Authenticates with Google using OAuth2."""
        if self.token_path.exists():
            self.creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Error refreshing token: {e}")
                    self.creds = None

            if not self.creds:
                if not self.credentials_path.exists():
                    logger.error("credentials.json not found!")
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
            self.service = build('keep', 'v1', credentials=self.creds)
            return True
        except Exception as e:
            logger.error(f"Failed to build service: {e}")
            return False

    def sync(self):
        """Syncs notes from Google Keep."""
        if not self.service:
            if not self.authenticate():
                return
        
        logger.info("Syncing notes from Google Keep (Official API)...")
        
        try:
            # Note: The Keep API 'list' method might differ slightly from Drive.
            # We need to checking documentation or trial-and-error given the private API nature.
            # Official docs: https://developers.google.com/keep/api/reference/rest/v1/notes/list
            
            request = self.service.notes().list()
            while request is not None:
                response = request.execute()
                notes = response.get('notes', [])
                
                for note in notes:
                    self._save_note(note)
                
                request = self.service.notes().list_next(previous_request=request, previous_response=response)
                
        except Exception as e:
            logger.error(f"Error syncing notes: {e}")
            if "403" in str(e):
                logger.error("ACCESS DENIED: It seems the official Google Keep API is restricted to Enterprise accounts only.")

    def _save_note(self, note):
        # API returns distinct structure: 'title', 'body' (text or list)
        title = note.get('title', 'Untitled')
        if not title:
            title = f"Keep Note {note.get('name', 'unknown').split('/')[-1]}"
            
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title).strip()
        filename = f"{safe_title}.md"
        filepath = self.inbox_dir / filename

        content = self._note_to_markdown(note)
        
        # Check for meaningful content
        if not content.strip():
            return

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Saved: {filename}")

    def _note_to_markdown(self, note):
        lines = []
        
        title = note.get('title')
        if title:
            lines.append(f"# {title}")
        
        # Create/Update time available? 
        # API v1 resource has 'createTime', 'updateTime'
        if 'createTime' in note:
             lines.append(f"**Created:** {note['createTime']}")
        if 'updateTime' in note:
             lines.append(f"**Updated:** {note['updateTime']}")
        
        lines.append(f"**ID:** {note.get('name')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        body = note.get('body', {})
        
        # Text Content
        if 'text' in body:
            lines.append(body['text'].get('text', ''))
            lines.append("")
            
        # List Content
        if 'list' in body:
            for item in body['list'].get('listItems', []):
                checked = "x" if item.get('checked') else " "
                text_content = item.get('text', {}).get('text', '')
                lines.append(f"- [{checked}] {text_content}")
            lines.append("")
            
        # Attachments
        if 'attachments' in note:
            lines.append("---")
            lines.append("**Attachments:**")
            for att in note['attachments']:
                lines.append(f"- {att.get('name', 'Unknown Attachment')} ({att.get('mimeType')})")

        return "\n".join(lines)
