import os
import frontmatter
import shutil
from pathlib import Path
from datetime import datetime
import logging

from core.services.state_manager import StateManager

# Assuming ProjectConfig is available or can be mocked
try:
    from config import ProjectConfig
except ImportError:
    class ProjectConfig:
        OBSIDIAN_VAULT = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./")) # Default to current dir if not set

logger = logging.getLogger("BrainOS.ActionProcessor")

def process_single_action_note(file_path: Path) -> bool:
    """
    Processes a single action note file using StateManager.
    Returns True if successfully processed and moved, False otherwise.
    """
    inbox_path = file_path.parent 
    archive_path = inbox_path / "Archive" / "Commands"
    errors_path = inbox_path / "_ERRORS"

    archive_path.mkdir(parents=True, exist_ok=True)
    errors_path.mkdir(parents=True, exist_ok=True)

    state_manager = StateManager()

    logger.info(f"Processing single action note: {file_path.name}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)

        if post.metadata.get('type') == 'command':
            action_type = post.metadata.get('action')
            items = post.metadata.get('items', [])

            if action_type in ['consume', 'add', 'buy', 'set']:
                success_all_items = True
                
                for item in items:
                    product_name = item.get('product')
                    qty = item.get('qty')
                    unit = item.get('unit', 'szt')
                    
                    if product_name and isinstance(qty, (int, float)):
                        try:
                            state_manager.update_inventory(product_name, qty, action_type, unit)
                        except Exception as e:
                            logger.error(f"Failed to update inventory for {product_name}: {e}")
                            success_all_items = False
                            break
                    else:
                        logger.warning(f"Invalid item data in command: {item}")
                        success_all_items = False
                        break

                if success_all_items:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    new_file_name = f"{file_path.stem}_{timestamp}.md"
                    shutil.move(str(file_path), str(archive_path / new_file_name))
                    logger.info(f"✅ Successfully processed and archived {file_path.name}")
                    return True
                else:
                    shutil.move(str(file_path), str(errors_path / file_path.name))
                    logger.error(f"❌ Failed to process {file_path.name}. Moved to _ERRORS.")
                    return False
            else:
                logger.warning(f"⚠️ Unknown action type '{action_type}' in {file_path.name}. Moving to _ERRORS.")
                shutil.move(str(file_path), str(errors_path / file_path.name))
                return False
        else:
            logger.info(f"ℹ️ {file_path.name} is not a command note.")
            return False

    except Exception as e:
        logger.error(f"An error occurred while processing {file_path.name}: {e}")
        shutil.move(str(file_path), str(errors_path / file_path.name))
        return False

def process_action_notes():
    """
    Scans the inbox for action notes.
    """
    inbox_path = ProjectConfig.OBSIDIAN_VAULT / "00_Inbox" # Use configured INBOX path
    
    if not inbox_path.exists():
         inbox_path = ProjectConfig.OBSIDIAN_VAULT / "inbox" # Fallback

    logger.info(f"Scanning for action notes in {inbox_path}...")

    for file_path in inbox_path.glob("*.md"):
        process_single_action_note(file_path)

if __name__ == "__main__":
    process_action_notes()
