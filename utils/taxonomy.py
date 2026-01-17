import json
import logging
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Set
from rapidfuzz import process, fuzz
import ollama

try:
    from config import ProjectConfig
    OLLAMA_MODEL = ProjectConfig.OLLAMA_MODEL
except ImportError:
    OLLAMA_MODEL = "SpeakLeash/bielik-11b-v2.3-instruct:Q5_K_M"

logger = logging.getLogger("TaxonomyGuard")
from adapters.google.gemini_adapter import UniversalBrain

class TaxonomyGuard:
    ALLOWED_UNITS = {
        "szt", "kg", "g", "l", "ml", "opakowanie", "słoik", "puszka", "porcja"
    }
    
    UNIT_MAPPING = {
        "szt.": "szt", "st": "szt", "sztuka": "szt", "sztuki": "szt",
        "kg.": "kg", "kilogram": "kg", "kilogramy": "kg",
        "gram": "g", "gramy": "g", "gramów": "g", "gr": "g",
        "litr": "l", "litry": "l", "ltr": "l",
        "opakowanie": "opakowanie", "opak": "opakowanie", "opakowania": "opakowanie", "op.": "opakowanie",
        "sloik": "słoik", "sloiki": "słoik", "słoiki": "słoik",
        "puszki": "puszka", "pusz": "puszka"
    }

    def __init__(self, json_path: str = "config/product_taxonomy.json"):
        self.json_path = Path(json_path)
        self.data = self._load_data()
        
        # 1. Budowanie szybkich indeksów
        self.categories = set(self.data.get("categories", {}).keys())
        self.canonical_products = set() 
        self.ocr_map = {} 
        
        self._build_indexes()
        self._check_integrity()

    def normalize_unit(self, unit: str) -> str:
        """Sprowadza jednostkę do formy kanonicznej."""
        if not unit: return "szt"
        u = unit.lower().strip().replace(" ", "")
        
        # 1. Bezpośrednie mapowanie
        if u in self.UNIT_MAPPING:
            return self.UNIT_MAPPING[u]
        
        # 2. Sprawdzenie czy już jest OK
        if u in self.ALLOWED_UNITS:
            return u
            
        # 3. Rozmyte dopasowanie (jeśli mapa zawiedzie)
        match = process.extractOne(u, list(self.ALLOWED_UNITS), scorer=fuzz.ratio)
        if match and match[1] > 85:
            return match[0]
            
        return "szt"

    def _load_data(self) -> Dict:
        if not self.json_path.exists(): return {"mappings": [], "categories": {}}
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception: return {"mappings": [], "categories": {}}

    def _build_indexes(self):
        mappings = self.data.get("mappings", [])
        for m in mappings:
            ocr_key = m['ocr'].strip().upper()
            self.ocr_map[ocr_key] = {
                "name": m['name'],
                "cat": m['cat'],
                "unit": self.normalize_unit(m.get('unit', 'szt'))
            }
            self.canonical_products.add(m['name'])
        
        self.ocr_patterns = list(self.ocr_map.keys())

    def _check_integrity(self):
        if not self.json_path.exists(): return
        sha = hashlib.sha256(self.json_path.read_bytes()).hexdigest()
        logger.info(f"Taxonomy Loaded. Hash: {sha[:8]}... Items: {len(self.ocr_map)}")

    def _llm_normalize_with_context(self, ocr_name: str, shop: str) -> Tuple[str, str, str]:
        candidates = process.extract(ocr_name, self.canonical_products, limit=5, scorer=fuzz.token_sort_ratio)
        candidates_str = ", ".join([f"'{c[0]}'" for c in candidates])
        cat_list_str = ", ".join(self.categories)
        unit_list_str = ", ".join(self.ALLOWED_UNITS)

        system_prompt = "Jesteś administratorem bazy danych produktów (MDM). Znormalizuj nazwę produktu z paragonu."
        user_prompt = f"""
        Sklep: {shop}.
        SUROWY OCR: "{ocr_name}"
        
        ZASADY:
        1. Jeśli produkt pasuje do jednego z ISTNIEJĄCYCH: [{candidates_str}], to UŻYJ TEJ DOKŁADNEJ NAZWY.
        2. Jeśli to nowy produkt, stwórz nową, ładną, ogólną nazwę po polsku (np. "Chleb Razowy" zamiast "Chleb Raz. 500g").
        3. Kategoria MUSI być jedną z: [{cat_list_str}]. Jeśli nie pasuje, użyj "INNE".
        4. Jednostka MUSI być jedną z: [{unit_list_str}].
        
        Odpowiedz TYLKO JSON: {{"name": "...", "cat": "...", "unit": "..."}}
        """
        
        try:
            # Note: Using the brain indirectly through OLLAMA_MODEL or config
            # But since we have UniversalBrain, we should ideally use it.
            brain = UniversalBrain()
            response_text = brain.generate_content(user_prompt, system_prompt=system_prompt, format="json")
            
            data = json.loads(response_text)
            cat = data.get('cat', 'INNE').upper()
            if cat not in self.categories:
                match = process.extractOne(cat, list(self.categories))
                cat = match[0] if match and match[1] > 80 else 'INNE'
            
            unit = self.normalize_unit(data.get('unit', 'szt'))
            return data.get('name', ocr_name), cat, unit
        except Exception as e:
            logger.error(f"LLM Norm failed: {e}")
            return ocr_name.title(), "INNE", "szt"

    def normalize_product(self, ocr_name: str, shop: str = "Sklep") -> Tuple[str, str, str]:
        ocr_clean = ocr_name.strip().upper()
        if ocr_clean in self.ocr_map:
            meta = self.ocr_map[ocr_clean]
            return meta['name'], meta['cat'], self.normalize_unit(meta['unit'])
        
        match = process.extractOne(ocr_clean, self.ocr_patterns, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 92:
            meta = self.ocr_map[match[0]]
            return meta['name'], meta['cat'], self.normalize_unit(meta['unit'])
            
        return self._llm_normalize_with_context(ocr_name, shop)