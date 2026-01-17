import os
import io
import re
import logging
import pdfplumber
import ollama
import json
import shutil
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from google.cloud import vision

from config import ProjectConfig, logger
from adapters.obsidian.vault_manager import ObsidianGardener
from utils.tag_engine import TagEngine
from utils.note_templates import (
    NoteBuilder, ReceiptNoteBuilder,
    escape_yaml_string, normalize_tag, sanitize_filename
)
from utils.receipt_agents import detect_shop

class PDFShredder:
    """
    Advanced PDF Processor: Extracts text and tables, detects compliance patterns,
    and generates structured Obsidian notes.
    """

    def __init__(self, vault_path: Optional[str] = None):
        self.vault_path = Path(vault_path) if vault_path else ProjectConfig.OBSIDIAN_VAULT
        self.output_dir = self.vault_path / "Compliance"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("PDFShredder")
        self.tag_engine = TagEngine()

        # Google Vision Setup
        if ProjectConfig.GOOGLE_APPLICATION_CREDENTIALS and ProjectConfig.GOOGLE_APPLICATION_CREDENTIALS.exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ProjectConfig.GOOGLE_APPLICATION_CREDENTIALS)
            self.vision_client = vision.ImageAnnotatorClient()
        else:
            self.vision_client = None
            self.logger.warning("Google Vision credentials not found. OCR will be disabled.")

    def detect_compliance_tags(self, text: str) -> List[str]:
        """Automated Tagging using Hybrid Tag Engine."""
        tags = self.tag_engine.generate_tags(text)
        return tags if tags else ["General"]

    def ocr_pdf_fallback(self, pdf_path: str) -> str:
        """OCR fallback using Google Vision for PDF files with no text layer."""
        if not self.vision_client:
            return ""
        
        self.logger.info(f"PDF has no text layer or very little text. Attempting Google Vision OCR: {pdf_path}")
        try:
            with open(pdf_path, 'rb') as f:
                content = f.read()
            
            input_config = vision.InputConfig(content=content, mime_type='application/pdf')
            feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
            
            # Process the first few pages
            request = vision.AnnotateFileRequest(
                input_config=input_config,
                features=[feature],
                pages=[1, 2, 3] 
            )
            
            response = self.vision_client.batch_annotate_files(requests=[request])
            
            texts = []
            for page_response in response.responses[0].responses:
                if page_response.full_text_annotation.text:
                    texts.append(page_response.full_text_annotation.text)
            
            return "\n".join(texts)
        except Exception as e:
            self.logger.error(f"OCR Fallback failed: {e}")
            return ""

    def extract_content(self, pdf_path: str) -> Tuple[str, List[str]]:
        """Extracts text and identifies compliance scope."""
        full_text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(page_text)
                    
                    # Optional: Table extraction logic (simplified for MVP)
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            full_text.append("| " + " | ".join([str(cell or "") for cell in table[0]]) + " |")
                            full_text.append("|" + "---|" * len(table[0]))
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {e}")

        combined_text = "\n".join(full_text).strip()
        
        # Fallback to Google Vision if no text was extracted or content is too short (possible scan)
        if not combined_text or len(combined_text) < 100:
            ocr_text = self.ocr_pdf_fallback(pdf_path)
            if ocr_text:
                combined_text = ocr_text
                self.logger.info("Successfully recovered text via Google Vision OCR.")

        tags = self.detect_compliance_tags(combined_text)
        return combined_text, tags

    def suggest_filename(self, text: str) -> str:
        """Uses LLM to suggest a standardized filename."""
        prompt = """
        Na podstawie treści dokumentu zaproponuj nazwę pliku w formacie: YYYY-MM-DD_Typ_Podmiot_Opis.
        Typ wybierz z listy: [Faktura, Paragon, Umowa, Wynik, Pismo, Prezentacja, Inne]. NIE używaj słowa "Typ" w nazwie.
        Podmiot: Nazwa firmy/osoby (np. Orange, UPC, LuxMed, Biedronka).
        Opis: Krótko (np. Internet, Krew, Prad, Zakupy).
        
        Jeśli nie znajdziesz daty w dokumencie, użyj dzisiejszej.
        Zwróć TYLKO nazwę pliku, bez rozszerzenia. NIE używaj nawiasów [ ].
        Przykład poprawnej nazwy: 2024-05-12_Paragon_Biedronka_Zakupy
        """
        try:
            response = ollama.chat(
                model=ProjectConfig.OLLAMA_MODEL_FAST,
                messages=[{'role': 'user', 'content': f"{prompt}\n\nTekst: {text[:2000]}"}]
            )
            filename = response['message']['content'].strip()
            # Sanitize - removing brackets and other problematic chars
            filename = filename.replace("[", "").replace("]", "").replace(" ", "_")
            filename = "".join(c for c in filename if c.isalnum() or c in ('-', '_')).strip()
            return filename
        except Exception as e:
            self.logger.error(f"Filename suggestion failed: {e}")
            return f"Doc_{{datetime.now().strftime('%Y%m%d_%H%M%S')}}"

    def extract_home_data(self, text: str) -> Dict[str, Any]:
        """Extracts structured data for Life Admin (invoices, etc.)."""
        prompt = """
        Przeanalizuj ten dokument. Jeśli to faktura lub dokument płatniczy, wyciągnij:
        - date (termin płatności YYYY-MM-DD)
        - amount (kwota z walutą)
        - account (numer konta)
        - subject (czego dotyczy)
        
        Zwróć JSON. Jeśli nie znaleziono, zwróć pusty JSON {{}}.
        """
        try:
            response = ollama.chat(
                model=ProjectConfig.OLLAMA_MODEL_FAST,
                messages=[{'role': 'user', 'content': f"{prompt}\n\nTekst: {text[:3000]}"}],
                format='json'
            )
            return json.loads(response['message']['content'])
        except Exception:
            return {}

    def preprocess_auchan(self, text: str) -> str:
        """Fixes Auchan specific OCR pattern (Product line followed by Qty x Price line)."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = line.strip()
            # Match pattern: 1 x4,48 4,48C or 0,864 x5,98 5,17C
            if re.match(r'^\d+([.,]\d+)?\s*x\d+([.,]\d+)?', line):
                 if cleaned:
                     cleaned[-1] = f"{cleaned[-1]} {line}"
                 else:
                     cleaned.append(line)
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def preprocess_ocr_text(self, text: str, shop: str = "Biedronka") -> str:
        """
        Zaawansowany preprocessing zależny od sklepu.
        """
        if shop == "Auchan":
            return self.preprocess_auchan(text)

        lines = text.split('\n')
        cleaned_lines = []
        
        for i in range(len(lines)):
            line = lines[i].strip()
            if not line: continue
            
            # 1. Łączenie linii typu 'Fil' i '.Makr...' (specyfika Biedronki)
            if line.startswith('.') and cleaned_lines:
                if len(cleaned_lines[-1]) <= 10: # Np. 'Fil'
                    cleaned_lines[-1] = cleaned_lines[-1] + line
                    continue

            # 2. Łączenie wiszących liter 'g', 'a', 'C' (końcówki gramatury/podatku)
            if len(line) == 1 and line.islower() and cleaned_lines:
                # Jeśli poprzednia linia to nazwa produktu, doklej
                if len(cleaned_lines[-1]) > 3:
                    cleaned_lines[-1] = cleaned_lines[-1] + line
                    continue
            
            # 3. Łączenie rozbitych zup Vifon (Vif... + 70g)
            if line.lower() == '70g' and cleaned_lines:
                cleaned_lines[-1] = cleaned_lines[-1] + line
                continue

            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines)


    def process_receipt_to_table(self, text: str) -> dict:
        """
        Używa LLM do zamiany OCR paragonu na strukturę danych.
        """
        detected_shop = detect_shop(text)
        text = self.preprocess_ocr_text(text, detected_shop)
        receipt_prompt = f"""
    Extract receipt data into JSON. This is a Polish {detected_shop} receipt.
    
    REQUIRED JSON STRUCTURE:
    {{
      "date": "YYYY-MM-DD",
      "shop": "{detected_shop}",
      "nip": "NIP number",
      "total_amount": 0.00,
      "table_markdown": "| Produkt | Ilość | Cena jedn. | Rabat | Suma | Data Ważności |\\n|---|---|---|---|---|---|\\n"
    }}

    RULES:
    1. **CONSOLIDATE**: Merge split product names.
    2. **PROMOTIONS**: Handle negative 'Rabat' or 'Upust'. 
       Suma for a row = (Quantity * Unit Price) + Rabat.
    3. **CLEANING**: Remove garbage chars.
    4. **AUCHAN**: Line '1 x4,48' means Quantity x Price.
    5. 'Data Ważności' column must be empty.
    """
        try:
            response = ollama.chat(
                model=ProjectConfig.OLLAMA_MODEL, 
                messages=[{'role': 'user', 'content': f"{receipt_prompt}\n\nTreść OCR po naprawie:\n{text[:5000]}"}],
                format='json'
            )
            data = json.loads(response['message']['content'])
            if 'table_markdown' not in data:
                data['table_markdown'] = "| Błąd: Brak tabeli |"
            return data
        except Exception as e:
            self.logger.error(f"Receipt extraction failed: {e}")
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "shop": detected_shop,
                "total_amount": 0.0,
                "table_markdown": "| Błąd generowania tabeli |"
            }

    def save_as_note(self, title: str, content: str, tags: List[str], home_data: Dict[str, Any] = None, attachment_path: str = None) -> Path:
        """Saves OCR content as a properly formatted Obsidian note."""

        # Detect if this is a receipt (Stricter Logic)
        text_lower = content.lower()
        title_lower = title.lower()

        # Keywords indicating a fiscal document
        fiscal_keywords = ['paragon fiskalny', 'faktura vat', 'faktura nr', 'sprzedaż opodatkowana', 'kwota do zapłaty', 'suma pln', 'nip']
        has_fiscal_keywords = any(k in text_lower for k in fiscal_keywords)

        # Keywords indicating a presentation/training (Negative check)
        presentation_keywords = ['slajd', 'agenda', 'szkolenie', 'prezentacja', 'spis treści', 'bibliografia', 'ćwiczenie', 'moduł']
        is_presentation = any(k in text_lower for k in presentation_keywords)

        # Shop names
        shop_names = ['jeronimo', 'biedronka', 'lidl', 'auchan', 'kaufland', 'rossmann', 'zabka', 'żabka']
        has_shop_name = any(s in text_lower for s in shop_names)

        # Force by filename prefix
        force_receipt = any(x in title_lower for x in ['paragon_', 'faktura_'])

        is_receipt = force_receipt or (
            not is_presentation and (
                (has_shop_name and has_fiscal_keywords)
                or (any(x in title_lower for x in ['paragon', 'faktura']) and has_fiscal_keywords)
                or ("FINANSE" in tags and has_fiscal_keywords)
            )
        )

        provider = detect_shop(content)
        safe_title = sanitize_filename(title)

        # --- BUILD NOTE USING TEMPLATES ---
        if is_receipt:
            output_dir = self.vault_path / "Paragony"
            output_dir.mkdir(parents=True, exist_ok=True)

            builder = ReceiptNoteBuilder(
                title=title,
                shop=provider,
                tags=tags + ['pdf-shredder']
            )

            # Add attachment embed
            if attachment_path:
                builder.add_embed(attachment_path)

            # Add OCR content
            builder.add_ocr_content(content[:20000])

            filepath = output_dir / f"{safe_title}.md"

        else:
            # Generic / Visual Note
            output_dir = self.vault_path / "00_Inbox" / "Notes"
            output_dir.mkdir(parents=True, exist_ok=True)

            builder = NoteBuilder(
                title=title,
                tags=tags + ['pdf-shredder', 'visual-note'],
                note_type="visual_note"
            )
            builder.add_field('provider', provider)
            builder.add_field('status', 'raw_ocr')

            # Add attachment embed
            if attachment_path:
                builder.add_embed(attachment_path)

            # Add callout about visual analysis
            builder.add_callout(
                'note',
                'Analiza Obrazu',
                'System wykrył tekst i etykiety wizualne.'
            )

            # Add OCR content section
            builder.add_section(
                'Rozpoznany Tekst',
                f"```\n{content[:20000]}\n```",
                icon='🔍'
            )

            filepath = output_dir / f"{safe_title}.md"

        # Write the note
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(builder.build())

        return filepath

    def process_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        content, tags = self.extract_content(pdf_path)
        if not content: return False, "Empty content"
        new_filename = self.suggest_filename(content)
        
        # FORCE PREFIX from Original Filename
        original_name = Path(pdf_path).name.lower()
        if "paragon_" in original_name and "paragon" not in new_filename.lower():
             new_filename = f"Paragon_{new_filename}"
        if "faktura_" in original_name and "faktura" not in new_filename.lower():
             new_filename = f"Faktura_{new_filename}"
        
        # Asset Management for PDF
        assets_dir = self.vault_path / "Assets"
        assets_dir.mkdir(exist_ok=True)
        saved_name = f"{new_filename}.pdf"
        shutil.copy2(pdf_path, assets_dir / saved_name)
        
        final_path = self.save_as_note(new_filename, content, tags, attachment_path=saved_name)
        
        ObsidianGardener(str(self.vault_path)).process_file(final_path)
        return True, str(final_path)

    def process_image(self, image_path: str) -> Tuple[bool, str]:
        # Logika process_image (uproszczona dla oszczędności miejsca, ale zachowująca label_detection)
        if not self.vision_client: return False, "No Vision API"
        try:
            with open(image_path, 'rb') as f:
                img_content = f.read()
            image = vision.Image(content=img_content)
            res = self.vision_client.annotate_image({'image': image, 'features': [{'type': 'TEXT_DETECTION'}, {'type': 'LABEL_DETECTION'}]})
            text = res.full_text_annotation.text if res.full_text_annotation else ""
            labels = [l.description for l in res.label_annotations]
            tags = self.detect_compliance_tags(text)
            tags.append("VisualNote")
            new_fn = self.suggest_filename(text)
            
            # FORCE PREFIX from Original Filename
            original_name = Path(image_path).name.lower()
            if "paragon_" in original_name and "paragon" not in new_fn.lower():
                 new_fn = f"Paragon_{new_fn}"
            if "faktura_" in original_name and "faktura" not in new_fn.lower():
                 new_fn = f"Faktura_{new_fn}"
            
            # Asset management
            assets_dir = self.vault_path / "Assets"
            assets_dir.mkdir(exist_ok=True)
            saved_name = f"{new_fn}{Path(image_path).suffix}"
            shutil.copy2(image_path, assets_dir / saved_name)
            
            final_path = self.save_as_note(new_fn, text, tags, attachment_path=saved_name)
            
            ObsidianGardener(str(self.vault_path)).process_file(final_path)
            return True, str(final_path)
        except Exception as e:
            return False, str(e)