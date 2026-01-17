from .base import BaseReceiptAgent
import re

class AuchanAgent(BaseReceiptAgent):
    def __init__(self):
        super().__init__("Auchan")

    def preprocess(self, text: str) -> str:
        # Agresywne łączenie linii: Produkt + Cena (często w nowej linii)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        merged = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Pattern ceny: 1 x4.48 lub 0.500 x10.00
            price_pattern = r'^\d+([.,]\d+)?\s*x\d+([.,]\d+)?'
            
            if i + 1 < len(lines) and re.match(price_pattern, lines[i+1]):
                merged.append(f"{line} {lines[i+1]}")
                i += 2
            elif i + 2 < len(lines) and re.match(price_pattern, lines[i+2]):
                merged.append(f"{line} {lines[i+2]}")
                i += 3
            else:
                merged.append(line)
                i += 1
        return "\n".join(merged)

    def get_prompt(self) -> str:
        return """
        Extract Auchan receipt data. 
        CRITICAL: Each product line is followed by a line like '1 x4.48'. 
        Merge them into one item. 
        
        OUTPUT JSON FORMAT:
        {
            "items": [
                {
                    "nazwa": "product name",
                    "ilosc": 1.0,
                    "cena_jedn": 10.99,
                    "suma": 10.99,
                    "rabat": 0.0
                }
            ]
        }
        """