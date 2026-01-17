import re
import json
import logging
from pathlib import Path

# Cache dla taksonomii, aby nie ładować pliku przy każdym wywołaniu
_TAXONOMY_CACHE = None

def _load_taxonomy():
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE
        
    try:
        # Zakładamy, że plik jest w katalogu głównym projektu, 3 poziomy wyżej
        # utils/receipt_agents/parser_utils.py -> utils/receipt_agents -> utils -> root
        taxonomy_path = Path(__file__).resolve().parent.parent.parent / "config/product_taxonomy.json"
        
        if taxonomy_path.exists():
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _TAXONOMY_CACHE = data.get("mappings", [])
                # Sortujemy mapowania od najdłuższych (najbardziej specyficznych) wzorców
                _TAXONOMY_CACHE.sort(key=lambda x: len(x['ocr']), reverse=True)
        else:
            logging.warning(f"Nie znaleziono pliku taksonomii: {taxonomy_path}")
            _TAXONOMY_CACHE = []
            
    except Exception as e:
        logging.error(f"Błąd ładowania taksonomii: {e}")
        _TAXONOMY_CACHE = []
        
    return _TAXONOMY_CACHE

def clean_raw_name_ocr(name: str) -> str:
    """
    Czyści surowy tekst z OCR: usuwa zbędne znaki, 
    normalizuje spacje i usuwa popularne końcówki jednostek.
    """
    if not name:
        return ""
    
    # 1. Usuwanie zbędnych znaków nie-alfanumerycznych na początku/końcu
    name = re.sub(r'^[^a-zA-Z0-9ĄĆĘŁŃÓŚŹŻąćęłńóśźż]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9ĄĆĘŁŃÓŚŹŻąćęłńóśźż]+$', '', name)
    
    # 2. Usuwanie wzorców typowych dla wag/ilości na końcu nazwy (np. "1kg", "x2", " 0.5")
    name = re.sub(r'\s+\d+[,.]?\d*\s*(kg|g|szt|l|ml|sztuk|op).*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+x\s*\d+.*$', '', name, flags=re.IGNORECASE)
    
    # 3. Usuwanie gwiazdek i innych śmieci OCR
    name = name.replace('*', '').replace('_', ' ').strip()
    
    # 4. Nadmiarowe spacje
    name = ' '.join(name.split())
    
    return name

def find_static_match(name: str) -> str:
    """
    Słownik mapowania nazw surowych na czyste nazwy systemowe.
    Korzysta z product_taxonomy.json.
    """
    name_upper = name.upper()
    mappings = _load_taxonomy()
    
    for item in mappings:
        if item['ocr'] in name_upper:
            return item['name']
            
    return None

def get_product_metadata(name: str) -> dict:
    """
    Zwraca domyślne metadane dla znanych kategorii produktów.
    TODO: Można to również zintegrować z taksonomią w przyszłości.
    """
    name_lower = name.lower()
    
    if any(x in name_lower for x in ['chleb', 'bułka', 'bagietka']):
        return {"kategoria": "Pieczywo", "jednostka": "szt"}
    if any(x in name_lower for x in ['mleko', 'ser', 'jogurt', 'masło', 'śmietana']):
        return {"kategoria": "Nabiał", "jednostka": "szt"}
    if any(x in name_lower for x in ['kurczak', 'szynka', 'kiełbasa', 'mięso', 'kabanos']):
        return {"kategoria": "Mięso", "jednostka": "kg"}
    if any(x in name_lower for x in ['woda', 'sok', 'cola', 'piwo']):
        return {"kategoria": "Napoje", "jednostka": "szt"}
    if any(x in name_lower for x in ['jabłko', 'banan', 'pomidor', 'ziemniak']):
        return {"kategoria": "Owoce i Warzywa", "jednostka": "kg"}
        
    return {"kategoria": "Inne", "jednostka": "szt"}