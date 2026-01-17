import os
import io
import shutil
import logging
import datetime
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config import ProjectConfig

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ProjectConfig.BASE_DIR / "backup.log")
    ]
)
logger = logging.getLogger("BackupManager")

SCOPES = ['https://www.googleapis.com/auth/drive.file']

class BackupManager:
    TARGET_FOLDER_NAME = "Obsidian Full Backups"
    MAX_BACKUPS = 4  # Keep last 4 weeks

    def __init__(self):
        self.creds = None
        self.service = None
        self.credentials_path = ProjectConfig.BASE_DIR / "credentials.json"
        self.token_path = ProjectConfig.BASE_DIR / "token_drive.json"
        self.vault_path = ProjectConfig.OBSIDIAN_VAULT

    def authenticate(self):
        """Authenticates using existing tokens or credentials."""
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
                    logger.error("Credentials file not found.")
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

    def _get_target_folder_id(self):
        query = f"mimeType='application/vnd.google-apps.folder' and name='{self.TARGET_FOLDER_NAME}' and trashed=false"
        results = self.service.files().list(q=query, spaces='drive', fields='files(id)').execute()
        files = results.get('files', [])

        if files:
            return files[0]['id']
        
        file_metadata = {
            'name': self.TARGET_FOLDER_NAME,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file = self.service.files().create(body=file_metadata, fields='id').execute()
        logger.info(f"Created backup folder: {self.TARGET_FOLDER_NAME}")
        return file.get('id')

    def create_zip_archive(self) -> Path:
        """Zips the vault directory."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        archive_name = f"obsidian_backup_{timestamp}"
        output_path = ProjectConfig.TEMP_DIR / archive_name
        
        logger.info(f"Zipping vault: {self.vault_path} -> {output_path}.zip")
        
        # shutil.make_archive creates .zip automatically
        zip_path = shutil.make_archive(
            str(output_path), 
            'zip', 
            str(self.vault_path)
        )
        return Path(zip_path)

    def cleanup_old_backups(self, folder_id):
        """Keeps only the latest MAX_BACKUPS files."""
        query = f"'{folder_id}' in parents and trashed=false"
        results = self.service.files().list(
            q=query, 
            fields='files(id, name, createdTime)', 
            orderBy='createdTime desc'
        ).execute()
        
        files = results.get('files', [])
        
        if len(files) > self.MAX_BACKUPS:
            to_delete = files[self.MAX_BACKUPS:]
            for f in to_delete:
                logger.info(f"Deleting old backup: {f['name']}")
                self.service.files().delete(fileId=f['id']).execute()

    def run_backup(self):
        if not self.authenticate():
            logger.error("Authentication failed. Aborting backup.")
            return

        folder_id = self._get_target_folder_id()
        if not folder_id:
            logger.error("Could not find/create backup folder.")
            return

        # 1. Create Zip
        try:
            zip_path = self.create_zip_archive()
        except Exception as e:
            logger.error(f"Failed to create zip: {e}")
            return

        # 2. Upload
        try:
            logger.info(f"Uploading {zip_path.name} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB)...")
            
            file_metadata = {
                'name': zip_path.name,
                'parents': [folder_id]
            }
            media = MediaFileUpload(
                str(zip_path), 
                mimetype='application/zip', 
                resumable=True
            )
            
            self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            logger.info("Upload complete.")
            
            # 3. Cleanup Cloud
            self.cleanup_old_backups(folder_id)

        except Exception as e:
            logger.error(f"Upload failed: {e}")
        finally:
            # 4. Cleanup Local
            if zip_path.exists():
                os.remove(zip_path)
                logger.info("Local temporary file removed.")

if __name__ == "__main__":
    BackupManager().run_backup()
