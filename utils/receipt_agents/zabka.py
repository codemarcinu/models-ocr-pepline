from .base import BaseReceiptAgent
import re


class ZabkaAgent(BaseReceiptAgent):
    """
    Agent for Żabka convenience store receipts.

    Żabka receipts typically have:
    - Compact format with abbreviated product names
    - Hot dog and coffee promotions
    - Loyalty program discounts (Żappka)
    """

    def __init__(self):
        super().__init__("Zabka")

    def preprocess(self, text: str) -> str:
        """
        Preprocess Żabka OCR text.

        Handles:
        - Żappka loyalty discounts
        - Hot food items (hot-dogi, kanapki)
        - Abbreviated product names
        """
        lines = text.split('\n')
        cleaned = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip Żappka loyalty program noise
            if any(skip in line.lower() for skip in ['żappka', 'zappka', 'punkty', 'punktów']):
                continue

            # Skip receipt footer noise
            if any(skip in line.lower() for skip in ['dziękujemy', 'zapraszamy', 'paragon fiskalny']):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    def get_prompt(self) -> str:
        return """
        Extract Żabka receipt data into JSON.
        Żabka is a convenience store with hot food (hot-dogi, kanapki).
        Handle Żappka loyalty discounts.
        Ignore footer messages and loyalty point info.

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
