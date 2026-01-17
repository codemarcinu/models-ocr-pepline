import sys
import os
from pathlib import Path
import pytest
import shutil

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from core.tools.receipt_cleaner import ReceiptSanitizer

# Sample receipt OCR text (Biedronka example)
SAMPLE_RECEIPT_OCR = """
JERONIMO MARTINS POLSKA S.A.
BIEDRONKA NR 1234
ul. Testowa 1, 00-001 Warszawa

Data: 13.01.2026    Godzina: 14:23
Paragon fiskalny: 123/456/2026

MLEKO 2% 1L              5.99 A
CHLEB RAZOWY             4.50 A
MASŁO EKSTRA 200G        8.99 A
JAJKA L 10SZT           12.99 A
SER GOUDA 250G           9.99 A
POMIDORY 1KG             7.50 A
BANANY 1KG               5.99 A

SUMA PLN                56.95
GOTÓWKA                 60.00
RESZTA                   3.05

NIP: 1234567890
Dziękujemy za zakupy!
"""

# Another receipt for cache testing (repeated items)
SAMPLE_RECEIPT_2_OCR = """
BIEDRONKA NR 1234
Data: 13.01.2026

MLEKO 2% 1L              5.99 A
CHLEB RAZOWY             4.50 A
MASŁO EKSTRA 200G        8.99 A
KAWA MIELONA            15.99 A
HERBATA CZARNA           8.50 A

SUMA PLN                43.97
"""

@pytest.fixture
def ocr_text():
    return SAMPLE_RECEIPT_OCR

@pytest.fixture
def ocr_text_2():
    return SAMPLE_RECEIPT_2_OCR

@pytest.fixture
def sanitizer(tmp_path):
    # Use tmp_path fixture provided by pytest for a temporary directory
    vault_path = tmp_path / "obsidian_vault_test"
    vault_path.mkdir()

    # Set test environment variable for async pipeline (if needed by ReceiptSanitizer)
    os.environ['ASYNC_PIPELINE'] = 'true'

    sanitizer_instance = ReceiptSanitizer(vault_path)
    yield sanitizer_instance

    # Teardown: Clean up the temporary vault if necessary
    if vault_path.exists():
        shutil.rmtree(vault_path, ignore_errors=True)
    
    # Clean up environment variable
    del os.environ['ASYNC_PIPELINE']