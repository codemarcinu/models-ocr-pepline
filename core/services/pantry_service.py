import os
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import os
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import hashlib
from pathlib import Path

# Assuming ProjectConfig is available or can be mocked
try:
    from config import ProjectConfig
except ImportError:
    class ProjectConfig:
        OBSIDIAN_VAULT = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./")) # Default to current dir if not set

from database.repositories.product_repo import ProductRepository
from adapters.obsidian.view_generator import MarkdownGenerator # New Import

class PantryService:
    def __init__(self, repo: ProductRepository, markdown_generator: MarkdownGenerator):
        self.repo = repo
        self.view_generator = markdown_generator # Initialize view generator

    def consume_product(self, name: str, qty: float) -> bool:
        """
        Records the consumption of a product.
        Checks for low stock after consumption.
        """
        product_info = self.repo.get_product_by_name(name)
        if not product_info:
            print(f"Product '{name}' not found, cannot consume.")
            return False
        
        # Use the product's default unit for consumption if not specified.
        # For simplicity, assuming consumption quantity always aligns with product's unit.
        self.repo.add_consumption(name, qty, product_info.jednostka_miary)

        # Check if stock is below minimum after consumption
        stock_and_min = self.repo.get_product_stock_and_min_level(name)
        if stock_and_min:
            current_stock, min_level = stock_and_min
            if current_stock < min_level:
                print(f"ALERT: Product '{name}' is below minimum stock level ({current_stock} {product_info.jednostka_miary} < {min_level} {product_info.jednostka_miary}). Consider adding to shopping list.")
        
        # Regenerate pantry view after successful modification
        self.view_generator.regenerate_pantry_view()
        return True

    def process_receipt(self, receipt_data: Dict) -> bool:
        """
        Processes a single receipt's data and saves it to the database.
        receipt_data should contain: 'shop_name', 'date', 'items' (list of dicts), 'source_file'.
        """
        shop_name = receipt_data.get('shop_name', 'Nieznany')
        date_str = receipt_data.get('date')
        items_raw = receipt_data.get('items', [])
        source_file = receipt_data.get('source_file', 'unknown')

        try:
            transaction_date = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else datetime.now().date()
        except ValueError:
            transaction_date = datetime.now().date()

        total_sum = sum(item.get('suma', 0.0) for item in items_raw)

        receipt_fingerprint = f"{transaction_date}|{shop_name}|{total_sum:.2f}"
        receipt_hash = hashlib.sha256(receipt_fingerprint.encode()).hexdigest()

        # Check if receipt already exists
        if not self.repo.get_unprocessed_receipt_hashes([receipt_hash]):
            print(f"Receipt with hash {receipt_hash} already exists. Skipping.")
            return False

        try:
            self.repo.save_transaction(
                shop_name=shop_name,
                transaction_date=transaction_date,
                total_sum=total_sum,
                receipt_hash=receipt_hash,
                source_file=source_file,
                items_data=items_raw
            )
            print(f"Successfully processed receipt from {shop_name} on {transaction_date}")
            # Regenerate pantry view after successful modification
            self.view_generator.regenerate_pantry_view()
            return True
        except Exception as e:
            print(f"Error processing receipt: {e}")
            return False

    def get_current_pantry_state(self) -> List[Dict]:
        """
        Retrieves the current state of the pantry from the repository.
        """
        return self.repo.get_pantry_state()

    def bulk_consume_products(self, consumption_list: List[Tuple[str, float]]) -> bool:
        """
        Records the consumption of multiple products in a single transaction.
        consumption_list: list of tuples (product_name, quantity)
        """
        product_data_for_bulk = []
        for product_name, quantity in consumption_list:
            if quantity <= 0:
                continue
            produkt = self.repo.get_product_by_name(product_name)
            if produkt:
                product_data_for_bulk.append((produkt.id, quantity, produkt.jednostka_miary))
        
        if product_data_for_bulk:
            try:
                success = self.repo.add_consumptions_bulk(product_data_for_bulk)
                if success:
                    self.view_generator.regenerate_pantry_view()
                return success
            except Exception as e:
                print(f"Error during bulk consumption: {e}")
                return False
        return False

    def add_manual_product(self, category: str, product_name: str, quantity: float, unit: str) -> bool:
        """
        Manually adds a product to the pantry (e.g., from GUI or Obsidian tag).
        Creates a 'Manual' receipt entry to record the addition.
        """
        try:
            # 1. Ensure Product exists or create it
            produkt = self.repo.add_or_update_product(
                name=product_name.strip().title(),
                category=category,
                unit=unit,
                price=0.0, # Manual additions don't have a price per item usually
                supplier="Ręczne"
            )

            # 2. Create a "Manual" receipt entry
            today = datetime.now().date()
            # Generate a unique hash for manual entries
            receipt_hash_content = f"MANUAL-{today}-{produkt.id}-{product_name}-{datetime.now().timestamp()}"
            receipt_hash = hashlib.sha256(receipt_hash_content.encode()).hexdigest()

            # Prepare items_data for save_transaction
            items_data = [{
                'nazwa': produkt.nazwa,
                'kategoria': produkt.kategoria,
                'jednostka': produkt.jednostka_miary,
                'ilosc': quantity,
                'cena_jedn': 0.0,
                'rabat': 0.0,
                'suma': 0.0,
                'data_waznosci': None
            }]

            self.repo.save_transaction(
                shop_name="Ręczne Dodanie",
                transaction_date=today,
                total_sum=0.0, # Sum of manual items is 0 for price tracking purposes
                receipt_hash=receipt_hash,
                source_file="Manual Entry",
                items_data=items_data
            )
            self.view_generator.regenerate_pantry_view()
            print(f"Manual Addition: {product_name} ({quantity} {unit})")
            return True
        except Exception as e:
            print(f"Failed to add manual product: {e}")
            return False

    def generate_shopping_list_note(self):
        """Generates a shopping list note in Obsidian based on low stock items."""
        pantry_data = self.repo.get_pantry_state()
        
        to_buy = []
        for p in pantry_data:
            if p['stan'] < p['minimum_ilosc']:
                to_buy.append(f"- [ ] {p['nazwa']} (obecnie: {p['stan']:.2f} {p['jednostka_miary']}, min: {p['minimum_ilosc']})")

        if not to_buy:
            return "Spiżarnia jest pełna! Brak produktów poniżej limitu."

        md = f"# 📝 Lista Zakupów\n"
        md += f"> Wygenerowano automatycznie: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        md += "## 🛒 Do kupienia:\n"
        md += "\n".join(to_buy)
        md += "\n\n---\n#stats #shopping"

        target_path = ProjectConfig.OBSIDIAN_VAULT / "Zasoby" / "Lista Zakupów.md"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(md, encoding='utf-8')
        return f"Wygenerowano listę ({len(to_buy)} pozycji) do {target_path.name}"


