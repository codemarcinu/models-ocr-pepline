from .base import BaseReceiptAgent

class LidlAgent(BaseReceiptAgent):
    def __init__(self):
        super().__init__("Lidl")

    def preprocess(self, text: str) -> str:
        # Lidl ma zazwyczaj czysty OCR, usuwamy tylko śmieci
        return self.clean_lines(text)

    def get_prompt(self) -> str:
        return """
        Extract Lidl receipt data into JSON.
        Handle 'RABAT' lines correctly by subtracting them from product sum or listing separately.
        
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
