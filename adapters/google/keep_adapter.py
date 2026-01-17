import logging
import gkeepapi
from pathlib import Path
from config import ProjectConfig
import re

logger = logging.getLogger("KeepAdapter")

class KeepAdapter:
    def __init__(self):
        self.keep = gkeepapi.Keep()
        self.token_path = ProjectConfig.TOKEN_KEEP_FILE
        self.inbox_dir = ProjectConfig.INBOX_DIR / "Keep"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def login(self, username=None, password=None):
        """
        Authenticates with Google Keep.
        First tries to use a cached token.
        If that fails, uses username/password to get a new token.
        """
        success = False
        
        # 1. Try Resume (Token)
        if self.token_path.exists():
            try:
                with open(self.token_path, 'r') as f:
                    token = f.read().strip()
                self.keep.resume(username, token)
                logger.info("Successfully resumed session with token.")
                success = True
            except Exception as e:
                logger.warning(f"Failed to resume session: {e}")

        # 2. Try Login (Password)
        if not success and username and password:
            try:
                # Remove spaces from app password if present
                clean_password = password.replace(" ", "")
                self.keep.authenticate(username, clean_password)
                token = self.keep.getMasterToken()
                with open(self.token_path, 'w') as f:
                    f.write(token)
                logger.info("Successfully logged in and saved new token.")
                success = True
            except Exception as e:
                logger.error(f"Failed to login: {e}")
                
        return success

    def sync(self, label_name="Obsidian"):
        """
        Syncs notes with a specific label to Obsidian.
        """
        logger.info("Syncing with server...")
        self.keep.sync()
        
        label = self.keep.findLabel(label_name)
        if not label:
            logger.warning(f"Label '{label_name}' not found in Keep.")
            return

        notes = self.keep.find(labels=[label], trashed=False, archived=False)
        
        count = 0
        for note in notes:
            self._save_note(note)
            count += 1
            
        logger.info(f"Processed {count} notes from Keep.")

    def _save_note(self, note):
        title = note.title if note.title else f"Keep Note {note.id}"
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title).strip()
        filename = f"{safe_title}.md"
        filepath = self.inbox_dir / filename

        content = self._note_to_markdown(note)
        
        # Check if content changed (simple check)
        if filepath.exists():
            existing_content = filepath.read_text(encoding='utf-8', errors='ignore')
            if existing_content == content:
                return # Skip if identical

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Saved: {filename}")

    def _note_to_markdown(self, note):
        lines = []
        
        # Metadata
        lines.append(f"# {note.title}")
        lines.append(f"**Created:** {note.timestamps.created.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"**Updated:** {note.timestamps.updated.strftime('%Y-%m-%d %H:%M')}")
        
        tags = [l.name for l in note.labels.all()]
        tags.append("keep-import")
        lines.append(f"**Tags:** #{' #'.join(tags)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Body
        if note.text:
            lines.append(note.text)
            lines.append("")

        # Checklists
        if note.items:
            for item in note.items:
                checked = "x" if item.checked else " "
                lines.append(f"- [{checked}] {item.text}")
            lines.append("")
            
        # Attachments
        if note.blobs:
            lines.append("---")
            lines.append("**Attachments:**")
            for blob in note.blobs:
                lines.append(f"- {blob.name} ({blob.type})")

        return "\n".join(lines)
