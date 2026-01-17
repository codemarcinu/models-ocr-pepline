import os
import logging
from pathlib import Path
from adapters.obsidian.vault_manager import ObsidianGardener
from config import ProjectConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReprocessAll")

def main():
    gardener = ObsidianGardener()
    vault_path = ProjectConfig.OBSIDIAN_VAULT
    
    logger.info(f"Starting reprocessing of all notes in {vault_path}...")
    
    count = 0
    changed = 0
    
    for root, _, files in os.walk(vault_path):
        for file in files:
            if file.endswith(".md") and not file.startswith("000_MOC"):
                file_path = os.path.join(root, file)
                success, message = gardener.process_file(file_path)
                count += 1
                if "applied" in message:
                    changed += 1
                    logger.info(f"Processed {file}: {message}")
                
    logger.info(f"Finished. Total files: {count}, Files updated: {changed}")
    
    # Regenerate MOCs at the end
    logger.info("Regenerating MOCs...")
    gardener.generate_index()
    logger.info("MOCs regenerated.")

if __name__ == "__main__":
    main()
