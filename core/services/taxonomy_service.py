import json
import logging
import shutil
import sys
import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from config import ProjectConfig
    from receipt_bridge import Produkt
except ImportError:
    # Fallback lub error handling jeśli struktura projektu jest inna
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TaxonomyLearner")

class TaxonomyLearner:
    def __init__(self):
        self.json_path = getattr(ProjectConfig, 'BASE_DIR', Path('.')) / "config/product_taxonomy.json"
        self.db_url = getattr(ProjectConfig, 'RECEIPT_DB_URL', None)
        
    def backup_taxonomy(self):
        if not self.json_path.exists(): return
        backup_dir = self.json_path.parent / "backups" / "taxonomy"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(self.json_path, backup_dir / f"product_taxonomy_{timestamp}.json")

    def load_taxonomy(self) -> dict:
        if not self.json_path.exists(): return {"taxonomy_version": "1.0", "categories": {}, "mappings": []}
        with open(self.json_path, 'r', encoding='utf-8') as f: return json.load(f)

    def save_taxonomy(self, data: dict):
        data['taxonomy_version'] = datetime.now().strftime("%Y.%m.%d_LEARNED")
        if 'mappings' in data: data['mappings'].sort(key=lambda x: len(x['ocr']), reverse=True)
        with open(self.json_path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Zapisano zmiany w {self.json_path.name}")

    def learn(self):
        if not self.db_url:
            logger.error("Brak RECEIPT_DB_URL.")
            return
        engine = create_engine(self.db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            db_products = session.query(Produkt).all()
        finally:
            session.close()

        data = self.load_taxonomy()
        existing_mappings = data.get("mappings", [])
        known_canonical_names = {m['name'].upper() for m in existing_mappings}
        added_count = 0
        
        for p in db_products:
            if not p.nazwa: continue
            p_name_upper = p.nazwa.upper()
            if p_name_upper in known_canonical_names: continue
            
            existing_mappings.append({
                "ocr": p_name_upper,
                "name": p.nazwa,
                "cat": p.kategoria,
                "unit": p.jednostka_miary
            })
            known_canonical_names.add(p_name_upper)
            added_count += 1

        if added_count > 0:
            self.backup_taxonomy()
            data['mappings'] = existing_mappings
            self.save_taxonomy(data)
            logger.info(f"🚀 Dodano {added_count} nowych produktów.")

if __name__ == "__main__":
    TaxonomyLearner().learn()
