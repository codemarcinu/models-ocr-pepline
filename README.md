# Receipt Bench Environment

Środowisko testowe do dopracowywania logiki odczytu paragonów (OCR + AI).

## Struktura
- \sync_receipt_pipeline.py\: Główny plik przetwarzania.
- \utils/receipt_agents/\: Logika dla konkretnych sklepów (Biedronka, Lidl, etc.).
- \config/\: Konfiguracja (taxonomy, prompty).
- \dapters/\: Adaptery AI (UniversalBrain).

## Uruchomienie
1. Upewnij się, że masz Pythona 3.10+.
2. Przejdź do folderu:
   \ash
   cd ~/receipt-tests
   \3. Utwórz venv i zainstaluj zależności:
   \ash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   \4. Ustaw zmienne w \.env\ (skopiuj \.env.example\ lub utwórz własny).
5. Testowanie:
   Możesz użyć \erify_extraction.py\ jako bazy do testów.

## Cel
Dopracowanie promptów w \utils/receipt_agents/\ oraz logiki fuzzy matching w \sync_receipt_pipeline.py\ aby osiągnąć 100% skuteczności na modelach lokalnych (Bielik, DeepSeek).
