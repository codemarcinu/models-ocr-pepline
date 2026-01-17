import json
import logging
import shutil
import difflib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from config import ProjectConfig
from core.services.context_emitter import ContextEmitter

logger = logging.getLogger("BrainOS.StateManager")

class StateManager:
    """
    Central authority for System State (Pantry, Tasks).
    Manages JSON files as the Source of Truth.
    """
    
    def __init__(self):
        self.pantry_path = ProjectConfig.BASE_DIR / "data" / "pantry.json"
        self.tasks_path = ProjectConfig.BASE_DIR / "data" / "tasks.json"
        self.emitter = ContextEmitter()
        
        # Ensure data dir exists
        self.pantry_path.parent.mkdir(exist_ok=True, parents=True)

    def _backup_file(self, path: Path):
        """Creates a .bak copy of the file before modification."""
        if path.exists():
            try:
                shutil.copy(path, path.with_suffix(".json.bak"))
            except Exception as e:
                logger.error(f"Backup failed for {path}: {e}")

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Error in {path}: {e}. restoring backup?")
            return {}

    def _save_json(self, path: Path, data: Dict[str, Any]):
        self._backup_file(path)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            # Trigger Context Refresh on every save
            self.emitter.refresh()
            
        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")

    # --- Pantry Logic ---

    def normalize_product_name(self, name: str, existing_names: list[str]) -> str:
        """
        Fuzzy matches product name to existing inventory to avoid duplicates.
        e.g. "mleko" -> "Mleko".
        """
        matches = difflib.get_close_matches(name.lower(), [n.lower() for n in existing_names], n=1, cutoff=0.8)
        
        if matches:
            # Find original case
            for real_name in existing_names:
                if real_name.lower() == matches[0]:
                    return real_name
        return name.capitalize() # Default formatting

    def update_inventory(self, item_name: str, qty: float, action: str, unit: str = "szt"):
        """
        Updates pantry inventory.
        action: "add", "remove", "set"
        """
        data = self._load_json(self.pantry_path)
        items = data.get("items", [])
        
        # Get existing names for fuzzy match
        existing_names = [i["name"] for i in items]
        normalized_name = self.normalize_product_name(item_name, existing_names)
        
        # Find product
        product = next((i for i in items if i["name"] == normalized_name), None)
        
        if not product:
            if action == "remove":
                logger.warning(f"Cannot remove non-existent item: {normalized_name}")
                return
            
            # Create new product
            product = {
                "id": f"prod_{int(datetime.now().timestamp())}",
                "name": normalized_name,
                "category": "Inne", # Default
                "quantity": 0.0,
                "unit": unit,
                "min_threshold": 1.0,
                "last_updated": datetime.now().isoformat()
            }
            items.append(product)
            logger.info(f"Created new product: {normalized_name}")

        # Update Logic
        current_qty = float(product.get("quantity", 0))
        
        if action == "add" or action == "buy":
            product["quantity"] = current_qty + float(qty)
        elif action == "remove" or action == "consume":
            product["quantity"] = max(0.0, current_qty - float(qty))
        elif action == "set":
            product["quantity"] = float(qty)
            
        product["last_updated"] = datetime.now().isoformat()
        
        # Save
        data["items"] = items
        data["metadata"] = data.get("metadata", {})
        data["metadata"]["last_sync"] = datetime.now().isoformat()
        
        self._save_json(self.pantry_path, data)
        logger.info(f"Inventory Updated: {normalized_name} -> {product['quantity']}")

    # --- Task/Shopping Logic ---

    def add_task(self, text: str, tags: list[str] = None):
        if tags is None: tags = []
        
        data = self._load_json(self.tasks_path)
        tasks = data.get("tasks", [])
        
        new_task = {
            "id": f"task_{int(datetime.now().timestamp())}",
            "content": text,
            "status": "pending",
            "priority": "medium",
            "tags": tags,
            "created_at": datetime.now().isoformat(),
            "source": "ai_agent"
        }
        
        tasks.append(new_task)
        data["tasks"] = tasks
        
        self._save_json(self.tasks_path, data)
        logger.info(f"Task Added: {text}")

    def add_shopping_item(self, item_name: str, urgent: bool = False):
        data = self._load_json(self.tasks_path)
        shopping_list = data.get("shopping_list", [])
        
        # Check duplicate
        if any(i["item"].lower() == item_name.lower() for i in shopping_list):
            logger.info(f"Shopping item already exists: {item_name}")
            return

        shopping_list.append({
            "item": item_name,
            "urgent": urgent
        })
        
        data["shopping_list"] = shopping_list
        self._save_json(self.tasks_path, data)
        logger.info(f"Shopping List Added: {item_name}")
