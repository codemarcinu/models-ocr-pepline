from .base import BaseReceiptAgent
import re

class BiedronkaAgent(BaseReceiptAgent):
    def __init__(self):
        super().__init__("Biedronka")

    def preprocess(self, text: str) -> str:
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Łączenie rozbitych nazw (kropka na początku nowej linii)
            if line.startswith('.') and cleaned:
                cleaned[-1] = f"{cleaned[-1]}{line}"
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def get_prompt(self) -> str:
        return f"""
        Extract Biedronka receipt data into JSON. 
        Focus on merging product names that might be split.
        Ignore 'PTU' and 'Suma'.

        OUTPUT JSON FORMAT:
        {{
            "items": [
                {{
                    "nazwa": "product name",
                    "ilosc": 1.0,
                    "cena_jedn": 10.99,
                    "suma": 10.99,
                    "rabat": 0.0
                }}
            ]
        }}
        """

