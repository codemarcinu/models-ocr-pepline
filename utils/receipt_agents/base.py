import re
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseReceiptAgent(ABC):
    def __init__(self, shop_name: str):
        self.shop_name = shop_name
        self.logger = logging.getLogger(f"Agent_{shop_name}")

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Czyszczenie specyficzne dla sklepu."""
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        """Prompt LLM specyficzny dla układu paragonu."""
        pass

    def clean_lines(self, text: str) -> str:
        """Wspólne czyszczenie dla wszystkich."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return "\n".join(lines)

    def detect_dates(self, text: str) -> List[str]:
        """Szuka dat w tekście OCR i próbuje je zunifikować."""
        # Formaty: YYYY-MM-DD, DD.MM.YYYY, DD-MM-YYYY, YYYY.MM.DD
        patterns = [
            r'\b(\d{4})[-./](\d{2})[-./](\d{2})\b',  # 2024-05-12
            r'\b(\d{2})[-./](\d{2})[-./](\d{4})\b',  # 12.05.2024
        ]
        
        found = []
        for p in patterns:
            matches = re.findall(p, text)
            for m in matches:
                if len(m[0]) == 4: # YYYY, MM, DD
                    found.append(f"{m[0]}-{m[1]}-{m[2]}")
                else: # DD, MM, YYYY
                    found.append(f"{m[2]}-{m[1]}-{m[0]}")
        
        # Filtrujemy nierealne daty (np. rok 0000 lub miesiąc 13)
        valid_dates = []
        for d in set(found):
            y, m, day = map(int, d.split('-'))
            if 2000 <= y <= 2030 and 1 <= m <= 12 and 1 <= day <= 31:
                valid_dates.append(d)
        
        return sorted(valid_dates)
