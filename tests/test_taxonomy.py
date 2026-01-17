import pytest
from unittest.mock import MagicMock, patch, mock_open
import json
from pathlib import Path
from utils.taxonomy import TaxonomyGuard

class TestTaxonomyGuard:

    @pytest.fixture
    def mock_json_data(self):
        return {
            "categories": {
                "SPOZYWCZE": ["chleb", "mleko"],
                "CHEMIA": ["mydlo"]
            },
            "mappings": [
                {"ocr": "CHLEB RAZ", "name": "Chleb Razowy", "cat": "SPOZYWCZE", "unit": "szt"},
                {"ocr": "MYDLO BIALE", "name": "Mydło Białe", "cat": "CHEMIA", "unit": "szt"}
            ]
        }

    @patch("pathlib.Path.read_bytes")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_init_loads_data(self, mock_file, mock_exists, mock_read_bytes, mock_json_data):
        mock_exists.return_value = True
        mock_read_bytes.return_value = b"dummy content"
        mock_file.return_value.read.return_value = json.dumps(mock_json_data)
        
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_json_data)

        with patch("json.load", return_value=mock_json_data):
            guard = TaxonomyGuard("dummy.json")
        
        assert "CHLEB RAZ" in guard.ocr_map
        assert "SPOZYWCZE" in guard.categories
        assert len(guard.canonical_products) > 0

    @patch("utils.taxonomy.UniversalBrain.generate_content")
    def test_normalize_product_exact_match(self, mock_gen, mock_json_data):
        with patch("json.load", return_value=mock_json_data), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_bytes", return_value=b"dummy"), \
             patch("builtins.open"):
            guard = TaxonomyGuard("dummy.json")
            
        name, cat, unit = guard.normalize_product("CHLEB RAZ")
        assert name == "Chleb Razowy"
        assert cat == "SPOZYWCZE"
        assert unit == "szt"
        
        mock_gen.assert_not_called()

    @patch("utils.taxonomy.UniversalBrain.generate_content")
    def test_normalize_product_fuzzy_match(self, mock_gen, mock_json_data):
        with patch("json.load", return_value=mock_json_data), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_bytes", return_value=b"dummy"), \
             patch("builtins.open"):
            guard = TaxonomyGuard("dummy.json")
            
        name, cat, unit = guard.normalize_product("CHLEB RAZ.")
        assert name == "Chleb Razowy"
        mock_gen.assert_not_called()

    @patch("utils.taxonomy.UniversalBrain.generate_content")
    def test_normalize_product_llm_fallback(self, mock_gen, mock_json_data):
        with patch("json.load", return_value=mock_json_data), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_bytes", return_value=b"dummy"), \
             patch("builtins.open"):
            guard = TaxonomyGuard("dummy.json")
    
        mock_response = json.dumps({
            "name": "Nowy Produkt",
            "cat": "SPOZYWCZE",
            "unit": "kg"
        })
        mock_gen.return_value = mock_response
    
        name, cat, unit = guard.normalize_product("COŚ DZIWNEGO 123")
    
        assert name == "Nowy Produkt"
        assert cat == "SPOZYWCZE"
        assert unit == "kg"
        mock_gen.assert_called_once()