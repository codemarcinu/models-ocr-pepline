import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any

from config import ProjectConfig
from utils.path_resolver import PathResolver

logger = logging.getLogger("BrainOS.ContextEmitter")

class ContextEmitter:
    """
    Generates a consolidated context file for Gemini (GEMINI_CONTEXT.md).
    Reads truth from JSON state files and exports to Google Drive.
    """
    
    OUTPUT_FILENAME = "GEMINI_CONTEXT.md"
    OUTPUT_SUBDIR = "OUTPUT" # /mnt/g/Mój dysk/IronBrain/OUTPUT/
    
    STATIC_KITCHEN_CONTEXT = """
## 🍳 Kitchen Context
* **Tools Available**: Frytkownica beztłuszczowa (Air Fryer), blender Braun, patelnia wok, szybkowar.
* **Preferences**: Lubię kuchnię azjatycką i włoską. Unikam bardzo pikantnych potraw.
"""

    def __init__(self):
        self.pantry_file = ProjectConfig.BASE_DIR / "data" / "pantry.json"
        self.tasks_file = ProjectConfig.BASE_DIR / "data" / "tasks.json"

    def refresh(self) -> bool:
        """
        Regenerates the GEMINI_CONTEXT.md file and pushes it to Drive.
        Should be called whenever state changes.
        """
        logger.info("Refreshing Gemini Context...")
        
        try:
            # 1. Load Data
            pantry_data = self._load_json(self.pantry_file)
            tasks_data = self._load_json(self.tasks_file)
            
            # 2. Build Markdown
            content = self._build_markdown(pantry_data, tasks_data)
            
            # 3. Save Locally (Optional, for debugging/backup)
            local_context_path = ProjectConfig.TEMP_DIR / self.OUTPUT_FILENAME
            with open(local_context_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            # 4. Push to Drive
            drive_dest = PathResolver.get_drive_path(f"{self.OUTPUT_SUBDIR}/{self.OUTPUT_FILENAME}")
            
            if drive_dest:
                PathResolver.safe_copy(local_context_path, drive_dest)
                logger.info(f"✅ Context Synced to Drive: {drive_dest}")
                return True
            else:
                logger.warning("⚠️ Drive not available. Context saved locally only.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to refresh context: {e}")
            return False

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}

    def _build_markdown(self, pantry: Dict, tasks: Dict) -> str:
        lines = []
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        lines.append(f"# 🧠 IronBrain Current State")
        lines.append(f"> Generated: {timestamp}")
        lines.append("")
        
        # --- Pantry Section ---
        lines.append("## 📦 Pantry & Inventory")
        if pantry and "items" in pantry:
            lines.append("| Item | Qty | Unit | Category |")
            lines.append("|---|---|---|---|")
            for item in pantry.get("items", []):
                name = item.get("name", "Unknown")
                qty = item.get("quantity", 0)
                unit = item.get("unit", "")
                cat = item.get("category", "Other")
                lines.append(f"| {name} | {qty} | {unit} | {cat} |")
        else:
            lines.append("_Pantry is empty or data missing._")
        lines.append("")
        
        # --- Tasks Section ---
        lines.append("## ✅ Tasks & Shopping")
        
        # Shopping List
        lines.append("### 🛒 Shopping List")
        shopping = tasks.get("shopping_list", [])
        if shopping:
            for item in shopping:
                urgent = "🔥" if item.get("urgent") else ""
                lines.append(f"- [ ] {item.get('item')} {urgent}")
        else:
            lines.append("_No items needed._")
        lines.append("")
        
        # Tasks
        lines.append("### 📝 Active Tasks")
        task_list = tasks.get("tasks", [])
        if task_list:
            for task in task_list:
                status = task.get("status", "pending")
                if status == "pending":
                    icon = "priority_high" if task.get("priority") == "high" else ""
                    lines.append(f"- [ ] {task.get('content')} {icon}")
        else:
            lines.append("_No active tasks._")
        lines.append("")
        
        # --- Static Context ---
        lines.append(self.STATIC_KITCHEN_CONTEXT)
        
        return "\n".join(lines)
