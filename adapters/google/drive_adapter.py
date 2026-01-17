import os
import io
import json
import logging
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from sqlalchemy import create_engine, text

from config import ProjectConfig
from adapters.google.gemini_adapter import GeminiBrain
from utils.note_templates import NoteBuilder, TranscriptNoteBuilder
from core.tools.transcriber import VideoTranscriber

# Setup logger
logger = logging.getLogger("DriveBridge")

# Scopes for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

class DriveBridge:
    TARGET_FOLDER_NAME = "Obsidian Exports"
    IMPORT_FOLDER_NAME = "Obsidian_Imports"

    def __init__(self, vault_path: str = None):
        self.vault_path = Path(vault_path) if vault_path else ProjectConfig.OBSIDIAN_VAULT
        self.creds = None
        self.service = None
        self.gemini = GeminiBrain()
        self.credentials_path = ProjectConfig.BASE_DIR / "credentials.json"
        self.token_path = ProjectConfig.BASE_DIR / "token_drive.json"
        self.state_path = ProjectConfig.BASE_DIR / "drive_sync_state.json"
        self._cached_folder_id = None
        self._sync_state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load sync state: {e}")
        return {}

    def _save_state(self):
        try:
            with open(self.state_path, 'w') as f:
                json.dump(self._sync_state, f)
        except Exception as e:
            logger.error(f"Failed to save sync state: {e}")

    def authenticate(self):
        """Authenticates the user for Google Drive API."""
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
                    logger.error(f"Credentials file not found at {self.credentials_path}")
                    return False
                
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path), SCOPES)
                    self.creds = flow.run_local_server(port=0)
                except Exception as e:
                     logger.error(f"OAuth flow failed: {e}")
                     return False

            with open(self.token_path, 'w') as token:
                token.write(self.creds.to_json())

        try:
            self.service = build('drive', 'v3', credentials=self.creds)
            return True
        except Exception as e:
            logger.error(f"Failed to build service: {e}")
            return False

    # --- IMPORT LOGIC (Gemini -> Local) ---
    
    def check_remote_inbox(self):
        """Checks remote folder for new files from Gemini/Phone."""
        if not self.service: return

        # 1. Find or Create Import Folder
        folder_id = self._get_folder_id_by_name(self.IMPORT_FOLDER_NAME, create_if_missing=True)
        if not folder_id: return

        # 2. List files
        q_files = f"'{folder_id}' in parents and trashed=false"
        try:
            results = self.service.files().list(q=q_files, fields="files(id, name, mimeType)").execute()
            items = results.get('files', [])
        except Exception as e:
            logger.error(f"Failed to list remote inbox: {e}")
            return

        if not items: return

        logger.info(f"🔎 Found {len(items)} new files in '{self.IMPORT_FOLDER_NAME}'")

        # 3. Download and Process
        for item in items:
            self._download_and_cleanup(item)

    def _download_and_cleanup(self, item):
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mimeType']

        # Sanitize filename
        safe_name = f"Drive_{datetime.datetime.now().strftime('%H%M')}_{file_name}"

        file_ext = file_name.split('.')[-1].lower() if '.' in file_name else ''
        is_image = file_ext in ['jpg', 'jpeg', 'png', 'webp']

        # Improved audio detection: check both extension AND MIME type
        audio_extensions = ['m4a', 'mp3', 'wav', 'ogg', 'flac', 'webm', 'aac']
        audio_mime_types = ['audio/', 'video/mp4', 'video/webm']  # m4a often has video/mp4 mime
        is_audio = (
            file_ext in audio_extensions or
            any(mime_type.startswith(m) for m in audio_mime_types)
        )

        is_transcript = "Transcript" in file_name or "Transkrypcja" in file_name
        
        # Ensure extension for non-images, non-audio and non-already-extended text files
        known_text_exts = ('.md', '.txt', '.json', '.pdf', '.docx', '.csv', '.xlsx')
        if not is_image and not is_audio and not safe_name.lower().endswith(known_text_exts):
             safe_name += ".md"
        
        # Destination path depends on file type processing
        dest_path_initial = ProjectConfig.TEMP_DIR / safe_name # Download to temp for processing
        final_dest_path = None # Will be set after processing

        try:
            request = None
            
            # Handle Google Docs (must be exported)
            if mime_type == 'application/vnd.google-apps.document':
                request = self.service.files().export_media(fileId=file_id, mimeType='text/plain')
            # Handle regular files
            else:
                request = self.service.files().get_media(fileId=file_id)

            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

            # Save locally (initially to temp or inbox for non-audio/image)
            if is_audio:
                with open(dest_path_initial, "wb") as f:
                    f.write(fh.getbuffer())
                logger.info(f"📥 Downloaded audio to temp: {safe_name}")
            else:
                with open(ProjectConfig.INBOX_DIR / safe_name, "wb") as f: # Direct to inbox for other files
                    f.write(fh.getbuffer())
                final_dest_path = ProjectConfig.INBOX_DIR / safe_name
                logger.info(f"📥 Downloaded: {safe_name}")

            # 2. Post-processing with proper templates
            if is_image:
                logger.info(f"📸 Przetwarzanie obrazu: {safe_name}")
                description = self.gemini.analyze_image(final_dest_path, prompt_type="general") # Use final_dest_path
                if description:
                    md_filename = safe_name + ".md"
                    md_path = ProjectConfig.INBOX_DIR / md_filename
                    builder = NoteBuilder(
                        title=f"Obraz: {safe_name}",
                        tags=['visual-inbox', 'drive-import', 'image'],
                        note_type="visual_note"
                    )
                    builder.add_embed(safe_name)
                    builder.add_section('Analiza AI', description, icon='🤖')
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(builder.build())
                    logger.info(f"📝 Utworzono notatkę dla obrazu: {md_filename}")
                # Set final_dest_path for cleanup
                final_dest_path = ProjectConfig.INBOX_DIR / md_filename

            elif is_audio: # New audio processing block
                logger.info(f"🎙️ Przetwarzanie audio (transkrypcja): {safe_name} [MIME: {mime_type}]")
                try:
                    transcriber = VideoTranscriber()

                    # Process the downloaded audio file
                    transcript_json_path = transcriber.process_local_file(str(dest_path_initial))

                    if not transcript_json_path or not Path(transcript_json_path).exists():
                        raise ValueError("Transcription failed - no JSON output generated")

                    # Load the generated JSON transcript
                    with open(transcript_json_path, 'r', encoding='utf-8') as f:
                        transcript_data = json.load(f)

                    content = transcript_data.get('content', '')

                    if not content or len(content.strip()) < 10:
                        raise ValueError(f"Transcription empty or too short: {len(content)} chars")

                    builder = TranscriptNoteBuilder(
                        title=f"Notatka Głosowa: {Path(safe_name).stem}",
                        source_type="voice",
                        tags=['voice-note', 'drive-import', 'transcribed']
                    )
                    builder.add_field('source_file', safe_name)
                    builder.add_transcript(content, collapsible=False)

                    md_filename = f"{Path(safe_name).stem}.md"
                    md_path = ProjectConfig.INBOX_DIR / md_filename
                    with open(md_path, "w", encoding='utf-8') as f:
                        f.write(builder.build())
                    logger.info(f"📝 Utworzono notatkę transkrypcji: {md_path}")

                    # Cleanup temporary audio file
                    os.remove(dest_path_initial)
                    # Cleanup JSON file
                    try:
                        Path(transcript_json_path).unlink()
                    except Exception:
                        pass
                    # Set final_dest_path for cleanup (pointing to the MD note)
                    final_dest_path = md_path

                except Exception as e:
                    logger.error(f"❌ Audio transcription FAILED for {safe_name}: {e}")
                    # Cleanup temp file on failure
                    if dest_path_initial.exists():
                        os.remove(dest_path_initial)
                    # Do NOT create markdown file with binary content
                    # Do NOT delete from Drive so user can retry
                    return

            elif is_transcript:
                logger.info(f"🎙️ Wykryto notatkę głosową (pre-transkrybowana): {file_name}")
                content = fh.getvalue().decode('utf-8', errors='ignore')

                builder = TranscriptNoteBuilder(
                    title=f"Notatka Głosowa (Import)",
                    source_type="voice",
                    tags=['voice-note', 'pixel', 'drive-import']
                )
                builder.add_field('source', file_name)
                builder.add_transcript(content, collapsible=False)
                builder.set_raw_content(f"\n---\n_Oryginalny plik: {file_name}_")

                with open(ProjectConfig.INBOX_DIR / safe_name, "w", encoding='utf-8') as f: # Use INBOX_DIR here
                    f.write(builder.build())
                # Set final_dest_path for cleanup
                final_dest_path = ProjectConfig.INBOX_DIR / safe_name
            
            # Delete from Drive to avoid re-downloading - ONLY if processing was successful
            if final_dest_path and final_dest_path.exists(): # Only delete if a local file was successfully created
                self.service.files().delete(fileId=file_id).execute()
                logger.info(f"✅ Usunięto '{file_name}' z Google Drive po przetworzeniu.")
            else:
                logger.warning(f"❌ Nie usunięto '{file_name}' z Google Drive - przetwarzanie nie powiodło się.")
        except Exception as e:
            logger.error(f"Error processing item {item.get('name', 'unknown')}: {e}")

    def _get_folder_id_by_name(self, name: str, create_if_missing=False) -> Optional[str]:
        query = f"mimeType='application/vnd.google-apps.folder' and name='{name}' and trashed=false"
        try:
            results = self.service.files().list(q=query, spaces='drive', fields='files(id)').execute()
            files = results.get('files', [])
            if files:
                return files[0]['id']
            
            if create_if_missing:
                meta = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
                f = self.service.files().create(body=meta, fields='id').execute()
                logger.info(f"Created folder: {name}")
                return f.get('id')
        except Exception as e:
            logger.error(f"Error finding folder {name}: {e}")
        return None

    # --- EXPORT LOGIC (Local -> Drive) ---

    def scan_for_uploads(self):
        """Scans vault for #gdrive files."""
        if not self.service:
            if not self.authenticate(): return

        # 1. Standard Tag Scan
        for md_file in self.vault_path.rglob("*.md"):
            self._process_file(md_file)
        
        # 2. Daily Note Sync
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        daily_note_path = self.vault_path / "Daily" / f"{today_str}.md"
        if daily_note_path.exists():
            self._process_file(daily_note_path, force_sync=True)

        self._save_state()

    def _process_file(self, file_path: Path, force_sync: bool = False):
        try:
            current_mtime = file_path.stat().st_mtime
            rel_path = str(file_path.relative_to(self.vault_path))
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception: return

        if "#gdrive" not in content and not force_sync: return

        last_mtime = self._sync_state.get(rel_path, 0)
        if current_mtime <= last_mtime: return

        logger.info(f"Syncing file: {file_path.name}")
        title = file_path.stem
        content_to_upload = content.replace("#gdrive", "").strip()
        link = self._upload_to_drive(title, content_to_upload)
        
        if link:
            self._sync_state[rel_path] = current_mtime
            if "[Drive Link]" not in content:
                new_content = f"[Drive Link]({link})\n\n{content}"
                file_path.write_text(new_content, encoding='utf-8')

    def _upload_to_drive(self, title: str, content: str) -> Optional[str]:
        folder_id = self._get_folder_id_by_name(self.TARGET_FOLDER_NAME, create_if_missing=True)
        if not folder_id: return None

        query = f"name = '{title}' and '{folder_id}' in parents and trashed = false"
        file_id = None
        try:
            results = self.service.files().list(q=query, fields='files(id, webViewLink)').execute()
            if results.get('files'):
                file_id = results['files'][0]['id']
                web_link = results['files'][0]['webViewLink']
        except Exception: pass

        media = MediaIoBaseUpload(io.BytesIO(content.encode('utf-8')), mimetype='text/plain', resumable=True)

        try:
            if file_id:
                self.service.files().update(fileId=file_id, media_body=media).execute()
                return web_link
            else:
                meta = {'name': title, 'mimeType': 'application/vnd.google-apps.document', 'parents': [folder_id]}
                f = self.service.files().create(body=meta, media_body=media, fields='webViewLink').execute()
                return f.get('webViewLink')
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None

    NON_FOOD_CATEGORIES = ["Chemia", "Kosmetyki", "Dom", "Elektronika", "Inne", "Leki", "Ubrania", "Higiena"]

    def _generate_pantry_report(self):
        """Generates a markdown report of the pantry status from JSON State."""
        from core.services.context_emitter import ContextEmitter
        logger.info("Generating Gemini Context Report (Headless)...")
        
        emitter = ContextEmitter()
        success = emitter.refresh()
        
        if success:
             logger.info("✅ Gemini Context refreshed and synced to Drive.")
        else:
             logger.error("❌ Failed to refresh Gemini Context.")

    def run_forever(self, interval_seconds: int = 600):
        logger.info(f"Starting DriveBridge 2.0 (Bidirectional). Interval: {interval_seconds}s")
        
        while True:
            try:
                # 1. Download from Gemini (Phone -> PC)
                self.check_remote_inbox()

                # 2. Upload to Drive (PC -> Phone)
                self.scan_for_uploads()

            except Exception as e:
                logger.error(f"Error during loop: {e}")
            
            import time
            time.sleep(interval_seconds)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    bridge = DriveBridge()
    
    if "--service" in sys.argv:
        bridge.run_forever()
    else:
        # On manual run, force pantry report for testing
        bridge._generate_pantry_report()
        bridge.scan_for_uploads()
