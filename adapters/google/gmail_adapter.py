"""
Gmail Adapter - Technical layer for Gmail API communication.

Part of Hexagonal Architecture - isolates Gmail API from business logic.
Handles: fetching emails, archiving, labeling, trashing.
"""

import base64
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import ProjectConfig

logger = logging.getLogger("GmailAdapter")

SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.labels',
]


@dataclass
class EmailMessage:
    """Parsed email data structure."""
    id: str
    thread_id: str
    sender: str
    sender_email: str
    subject: str
    snippet: str
    date: datetime
    labels: List[str]
    has_attachments: bool = False
    body_preview: str = ""


class GmailAdapter:
    """
    Adapter for Gmail API communication.

    Follows the same pattern as DriveBridge and CalendarBridge.
    Uses separate token file (token_gmail.json) for Gmail-specific scopes.
    """

    # Label names for AI triage (will be created if not exist)
    AI_LABELS = {
        "AI_Processed": "Label for emails processed by AI",
        "AI_Trash": "Emails AI suggests to delete (review before deleting)",
        "AI_Newsletter": "Detected newsletters",
        "AI_Finance": "Detected financial emails (invoices, receipts)",
        "AI_Important": "AI-flagged important emails",
    }

    def __init__(self):
        self.creds: Optional[Credentials] = None
        self.service = None
        self.credentials_path = ProjectConfig.BASE_DIR / "credentials.json"
        self.token_path = ProjectConfig.BASE_DIR / "token_gmail.json"
        self._label_cache: Dict[str, str] = {}  # name -> id mapping

    def authenticate(self) -> bool:
        """Authenticate with Gmail API using OAuth 2.0."""
        if self.token_path.exists():
            self.creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Error refreshing Gmail token: {e}")
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
                    logger.error(f"Gmail OAuth flow failed: {e}")
                    return False

            # Save refreshed/new credentials
            with open(self.token_path, 'w') as token:
                token.write(self.creds.to_json())

        try:
            self.service = build('gmail', 'v1', credentials=self.creds)
            self._ensure_ai_labels()
            return True
        except Exception as e:
            logger.error(f"Failed to build Gmail service: {e}")
            return False

    def _ensure_ai_labels(self):
        """Create AI labels if they don't exist."""
        if not self.service:
            return

        try:
            results = self.service.users().labels().list(userId='me').execute()
            existing_labels = {l['name']: l['id'] for l in results.get('labels', [])}
            self._label_cache = existing_labels

            for label_name in self.AI_LABELS.keys():
                if label_name not in existing_labels:
                    label_body = {
                        'name': label_name,
                        'labelListVisibility': 'labelShow',
                        'messageListVisibility': 'show'
                    }
                    created = self.service.users().labels().create(
                        userId='me', body=label_body).execute()
                    self._label_cache[label_name] = created['id']
                    logger.info(f"Created Gmail label: {label_name}")
        except Exception as e:
            logger.error(f"Failed to ensure AI labels: {e}")

    def _get_label_id(self, label_name: str) -> Optional[str]:
        """Get label ID by name, using cache."""
        if label_name in self._label_cache:
            return self._label_cache[label_name]

        # Refresh cache
        try:
            results = self.service.users().labels().list(userId='me').execute()
            self._label_cache = {l['name']: l['id'] for l in results.get('labels', [])}
            return self._label_cache.get(label_name)
        except Exception as e:
            logger.error(f"Failed to get label ID for {label_name}: {e}")
            return None

    def fetch_unread_emails(self, limit: int = 20, skip_ai_processed: bool = True) -> List[EmailMessage]:
        """
        Fetch unread emails from inbox.

        Args:
            limit: Maximum number of emails to fetch
            skip_ai_processed: Skip emails already labeled with AI_Processed

        Returns:
            List of EmailMessage objects
        """
        if not self.service:
            if not self.authenticate():
                return []

        try:
            # Build query - inbox, unread, optionally skip AI processed
            query = "in:inbox is:unread"
            if skip_ai_processed:
                query += " -label:AI_Processed"

            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=limit
            ).execute()

            messages = results.get('messages', [])
            if not messages:
                logger.info("No unread emails found")
                return []

            email_data = []
            for msg in messages:
                parsed = self._fetch_and_parse_message(msg['id'])
                if parsed:
                    email_data.append(parsed)

            logger.info(f"Fetched {len(email_data)} unread emails")
            return email_data

        except Exception as e:
            logger.error(f"Failed to fetch emails: {e}")
            return []

    def _fetch_and_parse_message(self, msg_id: str) -> Optional[EmailMessage]:
        """Fetch full message and parse into EmailMessage."""
        try:
            msg = self.service.users().messages().get(
                userId='me',
                id=msg_id,
                format='full'
            ).execute()

            headers = {h['name'].lower(): h['value'] for h in msg['payload'].get('headers', [])}

            # Parse sender
            from_header = headers.get('from', '')
            sender_name = from_header
            sender_email = from_header
            if '<' in from_header and '>' in from_header:
                sender_name = from_header.split('<')[0].strip().strip('"')
                sender_email = from_header.split('<')[1].split('>')[0]

            # Parse date
            date_str = headers.get('date', '')
            try:
                # Handle common date formats
                from email.utils import parsedate_to_datetime
                msg_date = parsedate_to_datetime(date_str)
            except Exception:
                msg_date = datetime.now()

            # Check for attachments
            has_attachments = self._has_attachments(msg['payload'])

            # Get body preview (first 500 chars of text/plain or snippet)
            body_preview = self._extract_body_preview(msg['payload'], max_chars=500)
            if not body_preview:
                body_preview = msg.get('snippet', '')

            return EmailMessage(
                id=msg['id'],
                thread_id=msg['threadId'],
                sender=sender_name,
                sender_email=sender_email,
                subject=headers.get('subject', '(no subject)'),
                snippet=msg.get('snippet', ''),
                date=msg_date,
                labels=msg.get('labelIds', []),
                has_attachments=has_attachments,
                body_preview=body_preview
            )

        except Exception as e:
            logger.error(f"Failed to parse message {msg_id}: {e}")
            return None

    def _has_attachments(self, payload: Dict) -> bool:
        """Check if email has attachments."""
        if 'parts' in payload:
            for part in payload['parts']:
                if part.get('filename'):
                    return True
                if 'parts' in part:
                    if self._has_attachments(part):
                        return True
        return False

    def _extract_body_preview(self, payload: Dict, max_chars: int = 500) -> str:
        """Extract text body preview from email payload."""
        if payload.get('mimeType') == 'text/plain':
            data = payload.get('body', {}).get('data', '')
            if data:
                try:
                    decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    return decoded[:max_chars]
                except Exception:
                    pass

        if 'parts' in payload:
            for part in payload['parts']:
                text = self._extract_body_preview(part, max_chars)
                if text:
                    return text

        return ""

    def archive_email(self, msg_id: str) -> bool:
        """
        Archive email (remove from inbox but keep in All Mail).

        Args:
            msg_id: Gmail message ID

        Returns:
            True if successful
        """
        try:
            self.service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['INBOX']}
            ).execute()
            logger.info(f"Archived email {msg_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to archive {msg_id}: {e}")
            return False

    def add_label(self, msg_id: str, label_name: str) -> bool:
        """
        Add a label to email.

        Args:
            msg_id: Gmail message ID
            label_name: Label name (will be created if doesn't exist)

        Returns:
            True if successful
        """
        label_id = self._get_label_id(label_name)
        if not label_id:
            logger.error(f"Label not found: {label_name}")
            return False

        try:
            self.service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            logger.info(f"Added label {label_name} to {msg_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add label {label_name} to {msg_id}: {e}")
            return False

    def remove_label(self, msg_id: str, label_name: str) -> bool:
        """Remove a label from email."""
        label_id = self._get_label_id(label_name)
        if not label_id:
            return False

        try:
            self.service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': [label_id]}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to remove label {label_name} from {msg_id}: {e}")
            return False

    def trash_email(self, msg_id: str) -> bool:
        """
        Move email to trash.

        Args:
            msg_id: Gmail message ID

        Returns:
            True if successful
        """
        try:
            self.service.users().messages().trash(userId='me', id=msg_id).execute()
            logger.info(f"Trashed email {msg_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to trash {msg_id}: {e}")
            return False

    def mark_as_read(self, msg_id: str) -> bool:
        """Mark email as read."""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Failed to mark {msg_id} as read: {e}")
            return False

    def get_inbox_stats(self) -> Dict[str, int]:
        """Get inbox statistics."""
        if not self.service:
            if not self.authenticate():
                return {}

        try:
            # Total unread
            unread = self.service.users().messages().list(
                userId='me', q='in:inbox is:unread', maxResults=1
            ).execute()
            unread_count = unread.get('resultSizeEstimate', 0)

            # Total inbox
            inbox = self.service.users().messages().list(
                userId='me', q='in:inbox', maxResults=1
            ).execute()
            inbox_count = inbox.get('resultSizeEstimate', 0)

            # AI processed
            ai_processed = self.service.users().messages().list(
                userId='me', q='label:AI_Processed', maxResults=1
            ).execute()
            ai_count = ai_processed.get('resultSizeEstimate', 0)

            return {
                'inbox_total': inbox_count,
                'unread': unread_count,
                'ai_processed': ai_count,
                'pending_triage': max(0, unread_count)
            }

        except Exception as e:
            logger.error(f"Failed to get inbox stats: {e}")
            return {}

    def get_user_labels(self) -> List[Dict[str, str]]:
        """
        Get user-created labels (excluding system labels).

        Returns:
            List of dicts with 'name' and 'id' keys
        """
        if not self.service:
            if not self.authenticate():
                return []

        # System labels to exclude
        SYSTEM_LABELS = {
            'INBOX', 'SPAM', 'TRASH', 'UNREAD', 'STARRED', 'IMPORTANT',
            'SENT', 'DRAFT', 'CATEGORY_PERSONAL', 'CATEGORY_SOCIAL',
            'CATEGORY_PROMOTIONS', 'CATEGORY_UPDATES', 'CATEGORY_FORUMS',
            'CHAT', 'OPENED', 'SNOOZED'
        }

        try:
            results = self.service.users().labels().list(userId='me').execute()
            labels = results.get('labels', [])

            user_labels = []
            for label in labels:
                label_name = label.get('name', '')
                label_id = label.get('id', '')

                # Skip system labels
                if label_id in SYSTEM_LABELS or label_id.startswith('CATEGORY_'):
                    continue

                # Include user labels (type='user') and AI labels
                if label.get('type') == 'user' or label_name.startswith('AI_'):
                    user_labels.append({
                        'name': label_name,
                        'id': label_id
                    })

            logger.info(f"Found {len(user_labels)} user labels")
            return user_labels

        except Exception as e:
            logger.error(f"Failed to get user labels: {e}")
            return []

    def create_label(self, label_name: str) -> Optional[str]:
        """
        Create a new label.

        Args:
            label_name: Name for the new label

        Returns:
            Label ID if created successfully, None otherwise
        """
        if not self.service:
            if not self.authenticate():
                return None

        try:
            label_body = {
                'name': label_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            created = self.service.users().labels().create(
                userId='me', body=label_body).execute()
            label_id = created.get('id')
            self._label_cache[label_name] = label_id
            logger.info(f"Created label: {label_name}")
            return label_id
        except Exception as e:
            logger.error(f"Failed to create label {label_name}: {e}")
            return None
