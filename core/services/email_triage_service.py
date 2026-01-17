"""
Email Triage Service - Business logic for AI-powered email inbox management.

Architecture:
    1. FAST PATH: Rules + Sender Cache (instant, no AI)
    2. SLOW PATH: AI classification (only for unknown senders)
    3. Learn: Cache AI decisions per sender for future emails

Features:
- Dynamic labels from user's Gmail
- Sender cache for instant processing of known senders
- Whitelist/Blacklist rules
- Obsidian notes for important emails
- Dry run mode for safety
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Set

from adapters.google.gmail_adapter import GmailAdapter, EmailMessage
from core.services.llm_router import SmartRouter, TaskType
from config import ProjectConfig

logger = logging.getLogger("EmailTriageService")


class TriageAction(Enum):
    """Possible actions for email triage."""
    ARCHIVE = "archive"
    DELETE = "delete"
    LABEL = "label"
    KEEP = "keep"
    TASK = "task"
    SKIP = "skip"


@dataclass
class SenderRule:
    """Cached rule for a sender."""
    sender_pattern: str
    action: str
    label: Optional[str]
    created_at: str
    source: str  # 'ai', 'manual', 'whitelist', 'blacklist'
    hit_count: int = 0


@dataclass
class TriageDecision:
    """Result of triage decision."""
    action: TriageAction
    label: Optional[str]
    confidence: float
    reason: str
    source: str  # 'cache', 'rule', 'ai'
    create_note: bool = False
    note_priority: str = "normal"  # 'high', 'normal', 'low'


@dataclass
class TriageResult:
    """Result of processing a single email."""
    email: EmailMessage
    decision: TriageDecision
    executed: bool
    dry_run: bool


class EmailTriageService:
    """
    Service for AI-powered email inbox management with caching.

    Flow:
    1. Check whitelist -> SKIP
    2. Check blacklist -> DELETE
    3. Check sender cache -> Use cached action
    4. AI classification -> Execute + Cache result
    """

    RULES_FILE = "email_rules.json"

    # Patterns that always get notes in Obsidian
    IMPORTANT_PATTERNS = [
        r"faktura|invoice|rachunek",
        r"payment|płatność|przelew",
        r"deadline|termin",
        r"meeting|spotkanie|wizyta",
        r"urgent|pilne|asap",
        r"confirm|potwierdź|zatwierdź",
    ]

    def __init__(
        self,
        dry_run: bool = True,
        vault_path: Optional[Path] = None
    ):
        self.adapter = GmailAdapter()
        self.router = SmartRouter()
        self.dry_run = dry_run
        self.vault_path = vault_path or ProjectConfig.OBSIDIAN_VAULT
        self.rules_path = ProjectConfig.BASE_DIR / self.RULES_FILE

        # Load rules
        self._rules = self._load_rules()
        self._user_labels: List[str] = []
        self._important_patterns = [re.compile(p, re.IGNORECASE) for p in self.IMPORTANT_PATTERNS]

        self._stats = {
            "processed": 0,
            "from_cache": 0,
            "from_ai": 0,
            "notes_created": 0,
            "errors": 0,
        }

    def _load_rules(self) -> Dict[str, Any]:
        """Load sender rules from JSON file."""
        default = {
            "whitelist": [],
            "blacklist": [],
            "sender_cache": {},
            "updated_at": None
        }

        if not self.rules_path.exists():
            return default

        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure all keys exist
                for key in default:
                    if key not in data:
                        data[key] = default[key]
                return data
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            return default

    def _save_rules(self):
        """Save rules to JSON file."""
        self._rules["updated_at"] = datetime.now().isoformat()
        try:
            with open(self.rules_path, 'w', encoding='utf-8') as f:
                json.dump(self._rules, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save rules: {e}")

    def _load_user_labels(self):
        """Load user's Gmail labels."""
        if not self._user_labels:
            labels = self.adapter.get_user_labels()
            self._user_labels = [l['name'] for l in labels]
            logger.info(f"Loaded {len(self._user_labels)} user labels")

    def _normalize_sender(self, email: str) -> str:
        """Normalize sender email for cache lookup."""
        # Extract domain for broader matching
        if '@' in email:
            return email.lower().strip()
        return email.lower().strip()

    def _check_whitelist(self, sender_email: str) -> bool:
        """Check if sender matches whitelist."""
        for pattern in self._rules.get("whitelist", []):
            try:
                if re.match(pattern, sender_email, re.IGNORECASE):
                    return True
            except re.error:
                continue
        return False

    def _check_blacklist(self, sender_email: str) -> bool:
        """Check if sender matches blacklist."""
        for pattern in self._rules.get("blacklist", []):
            try:
                if re.match(pattern, sender_email, re.IGNORECASE):
                    return True
            except re.error:
                continue
        return False

    def _check_sender_cache(self, sender_email: str) -> Optional[SenderRule]:
        """Check if sender has cached rule."""
        normalized = self._normalize_sender(sender_email)
        cache = self._rules.get("sender_cache", {})

        # Exact match
        if normalized in cache:
            rule_data = cache[normalized]
            return SenderRule(**rule_data)

        # Domain match (e.g., *@newsletter.com)
        if '@' in normalized:
            domain = normalized.split('@')[1]
            domain_key = f"*@{domain}"
            if domain_key in cache:
                rule_data = cache[domain_key]
                return SenderRule(**rule_data)

        return None

    def _cache_sender_decision(self, sender_email: str, decision: TriageDecision):
        """Cache AI decision for sender."""
        normalized = self._normalize_sender(sender_email)

        rule = SenderRule(
            sender_pattern=normalized,
            action=decision.action.value,
            label=decision.label,
            created_at=datetime.now().isoformat(),
            source="ai",
            hit_count=0
        )

        self._rules["sender_cache"][normalized] = asdict(rule)
        self._save_rules()
        logger.info(f"Cached rule for {normalized}: {decision.action.value}")

    def _is_important_email(self, email: EmailMessage) -> bool:
        """Check if email matches important patterns."""
        text = f"{email.subject} {email.body_preview}".lower()
        return any(p.search(text) for p in self._important_patterns)

    def _build_triage_prompt(self, email: EmailMessage) -> str:
        """Build prompt for AI triage with user's labels."""
        self._load_user_labels()

        # Format labels for prompt
        labels_str = ", ".join(self._user_labels[:20]) if self._user_labels else "Brak etykiet"

        return f"""Przeanalizuj email i zdecyduj co zrobić.

EMAIL:
Od: {email.sender} <{email.sender_email}>
Temat: {email.subject}
Treść: {email.body_preview[:400] if email.body_preview else email.snippet}
Załączniki: {"Tak" if email.has_attachments else "Nie"}

DOSTĘPNE ETYKIETY UŻYTKOWNIKA:
{labels_str}

AKCJE:
- archive: Usuń z inbox, zachowaj w All Mail
- delete: Przenieś do AI_Trash (spam, promocje)
- label: Dodaj etykietę i zarchiwizuj
- keep: Zostaw w inbox (ważne)
- task: Utwórz notatkę w Obsidian (wymaga działania)

ODPOWIEDZ TYLKO JSON:
{{
    "action": "archive|delete|label|keep|task",
    "label": "nazwa etykiety z listy powyżej lub null",
    "confidence": 0.0-1.0,
    "reason": "krótkie uzasadnienie po polsku",
    "create_note": true/false,
    "note_priority": "high|normal|low"
}}"""

    def _parse_ai_response(self, response: str) -> Optional[TriageDecision]:
        """Parse AI response into TriageDecision."""
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())

            action_map = {
                "archive": TriageAction.ARCHIVE,
                "delete": TriageAction.DELETE,
                "label": TriageAction.LABEL,
                "keep": TriageAction.KEEP,
                "task": TriageAction.TASK,
            }
            action = action_map.get(data.get("action", "archive"), TriageAction.ARCHIVE)

            return TriageDecision(
                action=action,
                label=data.get("label"),
                confidence=float(data.get("confidence", 0.5)),
                reason=data.get("reason", "AI decision"),
                source="ai",
                create_note=data.get("create_note", False),
                note_priority=data.get("note_priority", "normal")
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse AI response: {e}")
            return None

    def classify_email(self, email: EmailMessage) -> TriageDecision:
        """
        Classify email using rules + cache + AI.

        Priority:
        1. Whitelist -> SKIP
        2. Blacklist -> DELETE
        3. Cache hit -> Use cached action
        4. AI -> Classify and cache
        """
        sender = email.sender_email

        # 1. Whitelist check
        if self._check_whitelist(sender):
            return TriageDecision(
                action=TriageAction.SKIP,
                label=None,
                confidence=1.0,
                reason="Whitelist",
                source="rule"
            )

        # 2. Blacklist check
        if self._check_blacklist(sender):
            return TriageDecision(
                action=TriageAction.DELETE,
                label="AI_Trash",
                confidence=1.0,
                reason="Blacklist",
                source="rule"
            )

        # 3. Cache check
        cached = self._check_sender_cache(sender)
        if cached:
            # Update hit count
            normalized = self._normalize_sender(sender)
            if normalized in self._rules["sender_cache"]:
                self._rules["sender_cache"][normalized]["hit_count"] += 1
                self._save_rules()

            self._stats["from_cache"] += 1

            # Check if this specific email needs a note
            needs_note = self._is_important_email(email)

            return TriageDecision(
                action=TriageAction(cached.action),
                label=cached.label,
                confidence=0.9,
                reason=f"Cache ({cached.hit_count + 1} hits)",
                source="cache",
                create_note=needs_note,
                note_priority="high" if needs_note else "normal"
            )

        # 4. AI classification
        prompt = self._build_triage_prompt(email)
        result = self.router.execute(
            task_type=TaskType.EMAIL_TRIAGE,
            prompt=prompt,
            format="json",
            timeout=30.0
        )

        self._stats["from_ai"] += 1

        if result.success and result.content:
            decision = self._parse_ai_response(result.content)
            if decision:
                # Override: always create note for important patterns
                if self._is_important_email(email):
                    decision.create_note = True
                    decision.note_priority = "high"

                # Cache the decision
                self._cache_sender_decision(sender, decision)
                return decision

        # Fallback
        return TriageDecision(
            action=TriageAction.ARCHIVE,
            label=None,
            confidence=0.3,
            reason="Fallback: AI error",
            source="fallback"
        )

    def _create_obsidian_note(self, email: EmailMessage, decision: TriageDecision):
        """Create Obsidian note for important email."""
        # Generate safe filename
        date_str = email.date.strftime('%Y-%m-%d')
        safe_subject = re.sub(r'[<>:"/\\|?*]', '', email.subject[:50])
        filename = f"Email_{date_str}_{safe_subject}.md"

        # Priority emoji
        priority_emoji = {
            "high": "🔴",
            "normal": "🟡",
            "low": "🟢"
        }.get(decision.note_priority, "🟡")

        # Build note content
        content = f"""---
type: email-action
source: gmail
sender: {email.sender_email}
date: {date_str}
priority: {decision.note_priority}
status: pending
tags:
  - email
  - inbox
  - {decision.label.lower().replace('_', '-') if decision.label else 'triage'}
---

# {priority_emoji} {email.subject}

## Info
| | |
|---|---|
| **Od** | {email.sender} |
| **Email** | `{email.sender_email}` |
| **Data** | {email.date.strftime('%Y-%m-%d %H:%M')} |
| **Załączniki** | {"Tak" if email.has_attachments else "Nie"} |
| **AI Decyzja** | {decision.action.value} ({decision.confidence:.0%}) |

## Treść
{email.body_preview if email.body_preview else email.snippet}

## Do zrobienia
- [ ] Przeczytać i odpowiedzieć
- [ ] {decision.reason}

---
*Utworzono automatycznie przez Gmail Agent*
"""

        # Save to inbox
        note_path = self.vault_path / "00_Inbox" / filename
        try:
            note_path.write_text(content, encoding='utf-8')
            logger.info(f"Created Obsidian note: {filename}")
            self._stats["notes_created"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to create note: {e}")
            return False

    def _execute_decision(self, email: EmailMessage, decision: TriageDecision) -> bool:
        """Execute triage decision."""
        action = decision.action

        # Create note if needed (before other actions)
        if decision.create_note:
            self._create_obsidian_note(email, decision)

        if action == TriageAction.SKIP:
            return True

        if action == TriageAction.KEEP:
            self.adapter.add_label(email.id, "AI_Processed")
            if decision.label:
                self.adapter.add_label(email.id, decision.label)
            return True

        if action == TriageAction.ARCHIVE:
            self.adapter.add_label(email.id, "AI_Processed")
            if decision.label:
                self.adapter.add_label(email.id, decision.label)
            self.adapter.archive_email(email.id)
            return True

        if action == TriageAction.DELETE:
            self.adapter.add_label(email.id, "AI_Trash")
            self.adapter.add_label(email.id, "AI_Processed")
            self.adapter.archive_email(email.id)
            return True

        if action == TriageAction.LABEL:
            if decision.label:
                self.adapter.add_label(email.id, decision.label)
            self.adapter.add_label(email.id, "AI_Processed")
            self.adapter.archive_email(email.id)
            return True

        if action == TriageAction.TASK:
            # Note already created above
            self.adapter.add_label(email.id, "AI_Processed")
            if decision.label:
                self.adapter.add_label(email.id, decision.label)
            self.adapter.archive_email(email.id)
            return True

        return False

    def process_inbox(
        self,
        limit: int = 20,
        dry_run: Optional[bool] = None
    ) -> List[TriageResult]:
        """Process unread emails."""
        is_dry_run = dry_run if dry_run is not None else self.dry_run
        results = []

        logger.info(f"Starting email triage (dry_run={is_dry_run}, limit={limit})")

        emails = self.adapter.fetch_unread_emails(limit=limit)
        if not emails:
            return results

        for email in emails:
            try:
                decision = self.classify_email(email)

                if is_dry_run:
                    logger.info(
                        f"[DRY RUN] {email.subject[:40]}... -> "
                        f"{decision.action.value} [{decision.source}] "
                        f"{'📝' if decision.create_note else ''}"
                    )
                    executed = False
                else:
                    executed = self._execute_decision(email, decision)

                results.append(TriageResult(
                    email=email,
                    decision=decision,
                    executed=executed,
                    dry_run=is_dry_run
                ))
                self._stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing {email.id}: {e}")
                self._stats["errors"] += 1

        return results

    # =========================================================================
    # RULES MANAGEMENT
    # =========================================================================

    def add_to_whitelist(self, pattern: str):
        """Add pattern to whitelist."""
        if pattern not in self._rules["whitelist"]:
            self._rules["whitelist"].append(pattern)
            self._save_rules()
            logger.info(f"Added to whitelist: {pattern}")

    def add_to_blacklist(self, pattern: str):
        """Add pattern to blacklist."""
        if pattern not in self._rules["blacklist"]:
            self._rules["blacklist"].append(pattern)
            self._save_rules()
            logger.info(f"Added to blacklist: {pattern}")

    def remove_from_whitelist(self, pattern: str):
        """Remove pattern from whitelist."""
        if pattern in self._rules["whitelist"]:
            self._rules["whitelist"].remove(pattern)
            self._save_rules()

    def remove_from_blacklist(self, pattern: str):
        """Remove pattern from blacklist."""
        if pattern in self._rules["blacklist"]:
            self._rules["blacklist"].remove(pattern)
            self._save_rules()

    def get_rules(self) -> Dict[str, Any]:
        """Get all rules."""
        return {
            "whitelist": self._rules.get("whitelist", []),
            "blacklist": self._rules.get("blacklist", []),
            "cached_senders": len(self._rules.get("sender_cache", {})),
            "updated_at": self._rules.get("updated_at")
        }

    def get_cached_senders(self) -> List[Dict]:
        """Get list of cached sender rules."""
        cache = self._rules.get("sender_cache", {})
        return [
            {
                "sender": k,
                "action": v.get("action"),
                "label": v.get("label"),
                "hits": v.get("hit_count", 0),
                "source": v.get("source")
            }
            for k, v in cache.items()
        ]

    def clear_sender_cache(self, sender: Optional[str] = None):
        """Clear sender cache (all or specific sender)."""
        if sender:
            normalized = self._normalize_sender(sender)
            if normalized in self._rules["sender_cache"]:
                del self._rules["sender_cache"][normalized]
        else:
            self._rules["sender_cache"] = {}
        self._save_rules()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        inbox_stats = self.adapter.get_inbox_stats()
        return {
            **self._stats,
            **inbox_stats,
            "dry_run_mode": self.dry_run,
            "cached_senders": len(self._rules.get("sender_cache", {})),
            "whitelist_count": len(self._rules.get("whitelist", [])),
            "blacklist_count": len(self._rules.get("blacklist", [])),
        }

    def review_ai_trash(self) -> List[EmailMessage]:
        """Get emails in AI_Trash."""
        if not self.adapter.service:
            if not self.adapter.authenticate():
                return []

        try:
            results = self.adapter.service.users().messages().list(
                userId='me',
                q='label:AI_Trash',
                maxResults=50
            ).execute()

            messages = results.get('messages', [])
            emails = []
            for msg in messages:
                parsed = self.adapter._fetch_and_parse_message(msg['id'])
                if parsed:
                    emails.append(parsed)
            return emails
        except Exception as e:
            logger.error(f"Failed to get AI_Trash: {e}")
            return []

    def confirm_trash(self, msg_ids: Optional[List[str]] = None) -> int:
        """Permanently delete AI_Trash emails."""
        if self.dry_run:
            logger.warning("Cannot confirm trash in dry_run mode")
            return 0

        if not msg_ids:
            emails = self.review_ai_trash()
            msg_ids = [e.id for e in emails]

        deleted = 0
        for msg_id in msg_ids:
            if self.adapter.trash_email(msg_id):
                deleted += 1

        return deleted
