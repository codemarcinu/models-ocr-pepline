import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime, date
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import ProjectConfig
from utils.taxonomy import TaxonomyGuard
from utils.gpu_lock import GPULock
from adapters.google.gemini_adapter import UniversalBrain

# Import centralized shop detection and agents
from utils.receipt_agents import detect_shop, get_agent

# Phase 1: Async Receipt Pipeline (feature flag controlled)
try:
    from async_receipt_pipeline import AsyncReceiptPipeline
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

class ReceiptSanitizer:
    def __init__(self, vault_path: Path):
        self.vault_path = vault_path
        self.target_folders = ["Compliance", "Paragony", "Finanse"]
        self.logger = logging.getLogger("ReceiptSanitizer")

        taxonomy_path = Path(__file__).parent / "config/product_taxonomy.json"
        self.taxonomy = TaxonomyGuard(str(taxonomy_path))

        self.brain = UniversalBrain()

        # Initialize async pipeline if enabled
        self.use_async = os.getenv('ASYNC_PIPELINE', 'false').lower() == 'true'
        if self.use_async and ASYNC_AVAILABLE:
            self.async_pipeline = AsyncReceiptPipeline()
            self.logger.info("✨ Async pipeline enabled (Phase 1)")
        else:
            if self.use_async:
                self.logger.warning("ASYNC_PIPELINE=true but async_receipt_pipeline not available")
            self.use_async = False

    def generate_clean_data(self, ocr_text: str):
        """
        Generate cleaned receipt data from OCR text.

        Routes to async pipeline if enabled (Phase 1), otherwise uses
        original sync logic (backward compatible).

        Args:
            ocr_text: Raw OCR text from receipt

        Returns:
            Dict with 'items' list and optional metadata
        """
        shop = detect_shop(ocr_text)

        # Phase 1: Try async pipeline first (if enabled)
        if self.use_async:
            try:
                self.logger.info(f"Using async pipeline for {shop} receipt")
                result = self.async_pipeline.process_receipt_sync(ocr_text, shop)

                # Convert async format to original format
                return {
                    'items': result.get('items', []),
                    'date': result.get('date'),
                    'total': result.get('total_amount', 0),
                    'shop': result.get('shop', shop)
                }

            except Exception as e:
                self.logger.warning(
                    f"Async pipeline failed: {e}, falling back to sync"
                )
                # Fall through to original sync logic

        # Original sync logic (unchanged for backward compatibility)
        agent = get_agent(shop)
        cleaned_ocr = agent.preprocess(ocr_text)

        # --- LOGIC FIRST: Taxonomy & Regex ---
        lines = cleaned_ocr.split('\n')
        recognized_items = []
        unrecognized_lines = []
        
        # Regex for price/qty: usually looks like "1.00 x 5.00" or just "5.00" at the end
        price_pattern = r'(\d+[.,]\d{2})'
        qty_pattern = r'(\d+(?:[.,]\d{1,3})?)\s*[x*]'
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            match = process.extractOne(line.upper(), self.taxonomy.ocr_patterns, scorer=fuzz.partial_ratio)
            
            if match and match[1] >= 90:
                meta = self.taxonomy.ocr_map[match[0]]
                prices = re.findall(price_pattern, line)
                qty_match = re.search(qty_pattern, line)
                
                if prices:
                    qty = 1.0
                    if qty_match:
                         try:
                             qty = float(qty_match.group(1).replace(',', '.'))
                         except ValueError: pass

                    price = self._safe_float(prices[-1])
                    total = price
                    
                    if len(prices) >= 2:
                        price = self._safe_float(prices[-2])
                        total = self._safe_float(prices[-1])

                    recognized_items.append({
                        "name": meta['name'],
                        "qty": qty, 
                        "price": price,
                        "discount": 0.0,
                        "sum": total,
                        "category": meta['cat']
                    })
                    continue
            
            if any(char.isdigit() for char in line) or "SUMA" in line.upper() or "TOTAL" in line.upper():
                unrecognized_lines.append(line)

        # --- AI SECOND: UNIVERSAL BRAIN ---
        if not self.brain.available:
            self.logger.error(f"{self.brain.provider} brain not available for receipt cleaning.")
            return {"items": recognized_items}

        candidate_dates = agent.detect_dates(ocr_text)
        date_hint = f"Kandydaci dat z tekstu: {', '.join(candidate_dates)}" if candidate_dates else "Nie znaleziono."

        prompt_template = ProjectConfig.PROMPTS.get('receipt_cleaner', {}).get('master_prompt', "")
        system_prompt = agent.get_prompt()
        user_prompt = prompt_template.format(
            agent_prompt="", # Already in system_prompt
            date_hint=date_hint,
            ocr_text=cleaned_ocr[:12000] 
        ).strip()
        
        try:
            # Brain response usually contains markdown blocks, need to extract JSON
            response_text = self.brain.generate_content(user_prompt, system_prompt=system_prompt, format="json")
            if not response_text:
                return {"items": recognized_items}
                
            # Clean JSON from markdown if necessary
            json_text = response_text
            if "```json" in response_text:
                json_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
            elif "```" in response_text:
                json_text = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL).group(1)
            
            ai_data = json.loads(json_text)
            
            # Since we send full OCR, AI items ARE the items. 
            # We can still merge with recognized_items if we want to prioritize taxonomy,
            # but usually AI with full context is better.
            # Let's use AI results as primary but keep recognized if AI fails.
            ai_items = ai_data.get('items') or ai_data.get('produkty') or []
            
            # Filter out items that are already in recognized_items by name similarity?
            # Actually, let's just use AI items if available, as they have full context.
            if ai_items:
                return ai_data
            
            return {"items": recognized_items}
        except Exception as e:
            self.logger.error(f"AI Processing Error ({self.brain.provider}): {e}")
            # Fallback to recognized items with date if available
            if recognized_items:
                return {"items": recognized_items, "date": candidate_dates[0] if candidate_dates else None}
            return {"items": []}


    def _markdown_table_from_items(self, items, shop="Sklep"):
        md = "| Produkt | Ilość | Cena jedn. | Rabat | Suma | Kategoria |\n"
        md += "|---|---|---|---|---|---|\n"
        if not items: return md
        
        for item in items:
            # Flexible key mapping
            raw_name = item.get('name') or item.get('nazwa') or '???'
            qty = self._safe_float(item.get('qty') or item.get('quantity') or item.get('ilosc'), 1.0)
            price = self._safe_float(item.get('price') or item.get('unit_price') or item.get('cena'), 0.0)
            discount = item.get('discount') or item.get('rabat')
            if isinstance(discount, dict):
                discount = self._safe_float(discount.get('wartosc') or discount.get('value'), 0.0)
            else:
                discount = self._safe_float(discount, 0.0)
                
            suma = self._safe_float(item.get('sum') or item.get('total_price') or item.get('suma'), 0.0)
            
            std_name, category, unit = self.taxonomy.normalize_product(raw_name, shop)
            md += f"| {std_name} | {qty} {unit} | {price:.2f} | {discount:.2f} | {suma:.2f} | {category} |\n"
        return md

    def _safe_float(self, val, default=0.0):
        if val is None: return default
        try:
            if isinstance(val, str):
                # Handle cases like "-8.50" or "1 234,56"
                val = val.replace(',', '.').replace(' ', '').replace('zł', '').strip()
            return abs(float(val)) # Use absolute value for discounts if they are negative
        except (ValueError, TypeError):
            return default

    def _parse_date_str(self, date_input: any) -> str or None:
        """Tries to parse a date from various formats into YYYY-MM-DD."""
        if not date_input:
            return None
        
        if isinstance(date_input, (datetime, date)):
            return date_input.strftime('%Y-%m-%d')

        date_str = str(date_input)
        formats_to_try = [
            '%Y-%m-%d', '%d-%m-%Y', '%d.%m.%Y', '%Y.%m.%d', '%Y/%m/%d', '%d/%m/%Y'
        ]
        
        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_str[:10], fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
                
        self.logger.warning(f"Could not parse date: {date_str}")
        return None

    def sanitize_file(self, file_path: Path) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except: return False

        # Guard: Only process if tagged for verification
        if "to-verify" not in content:
             return False

        if "## 📜 Oryginalny OCR" not in content:
            return False
            
        # If it's already synced, skip
        if "#synced-db" in content or "synced-db" in content:
             return False

        ocr_match = re.search(r'## 📜 Oryginalny OCR\s*\n(.*?)(?=\n#|\Z)', content, re.DOTALL)
        if not ocr_match: return False
        ocr_text = ocr_match.group(1).strip()
        shop = detect_shop(ocr_text)

        data = self.generate_clean_data(ocr_text)
        if not data: return False

        # Support 'items' or 'produkty'
        items = data.get('items') or data.get('produkty') or []
        new_table = self._markdown_table_from_items(items, shop)
        
        # LOGIKA DATY (OCR -> AI -> Filename -> Today)
        parsed_date = self._parse_date_str(data.get('date') or data.get('data'))
        date_str = parsed_date or '1970-01-01'

        ocr_dates = get_agent(shop).detect_dates(ocr_text)
        today = datetime.now().strftime("%Y-%m-%d")

        # 1. Jeśli data z AI/Pipeline jest 'niewiarygodna' (dzisiejsza/przyszła/brak), a mamy datę z OCR twardego -> biorę OCR
        if (date_str == '1970-01-01' or date_str >= today) and ocr_dates:
            date_str = ocr_dates[0]

        # 2. Jeśli nadal brak dobrej daty (OCR też zawiódł), szukamy w nazwie pliku
        if date_str == '1970-01-01' or date_str >= today:
            filename_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_path.name)
            if filename_match:
                candidate_from_name = filename_match.group(1)
                if candidate_from_name <= today:
                    date_str = candidate_from_name
                    self.logger.info(f"Użyto daty z nazwy pliku: {date_str}")

        # 3. Jeśli wszystko zawiodło -> Dzisiaj
        if date_str == '1970-01-01':
            date_str = today

        data['date'] = date_str

        content = self._update_note_content(content, data, new_table, shop)
        
        # ORGANIZACJA
        year, month = date_str[:4], date_str[:7]
        target_dir = self.vault_path / "Paragony" / year / month
        target_dir.mkdir(parents=True, exist_ok=True)
        new_path = target_dir / file_path.name
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        if file_path.resolve() != new_path.resolve():
            try:
                os.rename(file_path, new_path)
                self.logger.info(f"Moved to {year}/{month}")
            except OSError: pass
        return True

    def _update_note_content(self, content, data, table, shop):
        # Extract total amount from various possible keys
        total_val = 0.0
        total_field = data.get('total') or data.get('total_amount') or data.get('suma')

        if isinstance(total_field, dict):
            total_val = self._safe_float(total_field.get('sum') or total_field.get('value'))
        elif total_field is not None:
            total_val = self._safe_float(total_field)
        
        total = total_val
        
        if "provider:" not in content:
            content = re.sub(r'(---\n)', f'\\1provider: {shop}\n', content, count=1)
        
        # Update or add total_amount in frontmatter
        if "total_amount:" in content:
            content = re.sub(r'total_amount: .*\n', f'total_amount: {total}\n', content)
        else:
            content = re.sub(r'(---\n)', f'\\1total_amount: {total}\n', content, count=1)
        
        # Table update - look for the specific table header
        table_pattern = r'\| Produkt \| Ilość \| Cena jedn\. \| Rabat \| Suma \| Kategoria \|\n\|---\|---\|---\|---\|---\|---\|\n(.*?)(?=\n\n|\Z|\n#)'
        if re.search(table_pattern, content, re.DOTALL):
            content = re.sub(table_pattern, table.strip(), content, flags=re.DOTALL)
        else:
            # Fallback if table header matches but content is different
            table_start = content.find("| Produkt |")
            if table_start != -1:
                rest = content[table_start:]
                table_end_match = re.search(r'\n\n|(?=\n#)', rest)
                if table_end_match:
                    content = content[:table_start] + table + content[table_start + table_end_match.start():]
            else:
                 header = "## 🛒 Weryfikacja Paragonu"
                 if header in content:
                     content = content.replace(header, f"{header}\n\n{table}")

        # JSON Embedding
        json_block = f"\n## 🛠️ Dane Strukturalne (JSON)\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```\n"
        if "## 🛠️ Dane Strukturalne" in content:
            content = re.sub(r'## 🛠️ Dane Strukturalne \(JSON\)\n```json\n.*?\n```', json_block.strip(), content, flags=re.DOTALL)
        else:
            ocr_header = "## 📜 Oryginalny OCR"
            if ocr_header in content:
                content = content.replace(ocr_header, f"{json_block}\n{ocr_header}")
            else:
                content += json_block

        # Robust Tag Update
        content = content.replace("#to-verify", "#manual-verify")
        content = content.replace("- to-verify", "- manual-verify")
        
        # Update status to waiting-for-user
        if "status: raw_ocr" in content:
            content = content.replace("status: raw_ocr", "status: waiting-for-user")
        
        return content


    def run_batch(self):
        for folder in self.target_folders:
            path = self.vault_path / folder
            if not path.exists(): continue
            for file_path in path.glob("**/*.md"):
                self.sanitize_file(file_path)

if __name__ == "__main__":
    sanitizer = ReceiptSanitizer(ProjectConfig.OBSIDIAN_VAULT)
    sanitizer.run_batch()
