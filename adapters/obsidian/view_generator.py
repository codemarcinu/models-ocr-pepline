import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Assuming ProjectConfig is available or can be mocked
try:
    from config import ProjectConfig
except ImportError:
    class ProjectConfig:
        OBSIDIAN_VAULT = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./")) # Default to current dir if not set

# We no longer import PantryService or ProductRepository directly here to break the circular dependency.
# Instead, these will be passed as arguments to regenerate_pantry_view if needed.

class MarkdownGenerator:
    def __init__(self):
        # No longer initializing PantryService or ProductRepository here
        pass

    def regenerate_pantry_view(self, pantry_service) -> None:
        """
        Generates the 'Spiżarnia.md' note in Obsidian based on the current pantry state from the database.
        This function now retrieves data from PantryService, which is passed as a dependency.
        """
        pantry_data = pantry_service.get_current_pantry_state()

        # Emoji dla kategorii (copied from original receipt_bridge for consistency)
        CAT_EMOJIS = {
            "NABIAŁ": "🥛", "MIĘSO_WĘDLINY": "🥩", "RYBY": "🐟",
            "WARZYWA_OWOCE": "🍎", "SPIŻARNIA": "🍝", "PIECZYWO": "🍞",
            "NAPOJE": "🧃", "DANIA_GOTOWE": "🍕", "PRZETWORY": "🥫",
            "ŚNIADANIE": "🥣", "SUCHE": "🍝", "DODATKI": "🧂",
            "PRZEKĄSKI": "🍿", "CHEMIA_HIGIENA": "🧼",
            "ELEKTRONIKA_DOM": "💡", "UŻYWKI": "🚬", "INNE": "📦"
        }

        if not pantry_data:
            md = "# 📦 Stan Spiżarni\n"
            md += "> [!WARNING] READ ONLY\n"
            md += "> Ten plik jest generowany automatycznie z bazy danych.\n"
            md += "> Wszelkie ręczne zmiany zostaną nadpisane.\n"
            md += "> Aby zmienić stan, użyj CLI lub utwórz notatkę akcji.\n\n"
            md += "Brak produktów w bazie."
        else:
            # Grouping by category
            by_category = {}
            for item in pantry_data:
                cat = item['kategoria'].upper() # Ensure upper case for emoji matching
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(item)

            # Statistics
            total_items = len(pantry_data)
            in_stock = sum(1 for i in pantry_data if i['stan'] > 0)
            low_stock = sum(1 for i in pantry_data if 0 < i['stan'] < i['minimum_ilosc'])
            out_of_stock = sum(1 for i in pantry_data if i['stan'] <= 0)

            # Build MD
            md = f"# 📦 Stan Spiżarni\n"
            md += "> [!WARNING] READ ONLY\n"
            md += "> Ten plik jest generowany automatycznie z bazy danych.\n"
            md += "> Wszelkie ręczne zmiany zostaną nadpisane.\n"
            md += "> Aby zmienić stan, użyj CLI lub utwórz notatkę akcji.\n\n"
            md += f"> Ostatnia aktualizacja: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            md += f"**Podsumowanie:** ✅ {in_stock} w magazynie | ⚠️ {low_stock} niski stan | ❌ {out_of_stock} brak\n\n"
            md += "---\n\n"

            # Main table grouped by category
            for cat in sorted(by_category.keys()):
                items = by_category[cat]
                emoji = CAT_EMOJIS.get(cat, "📦")

                md += f"## {emoji} {cat}\n\n"
                md += "| Produkt | Stan | Min | Jednostka | Status | Ost. cena |\n"
                md += "|---------|-----:|----:|-----------|:------:|----------:|\n"

                for item in sorted(items, key=lambda x: x['nazwa']):
                    stan = item['stan']
                    min_stan = item['minimum_ilosc']

                    if stan <= 0:
                        status = "❌"
                    elif stan < min_stan:
                        status = "⚠️"
                    else:
                        status = "✅"

                    cena_str = f"{item['cena_zakupu']:.2f} zł" if item['cena_zakupu'] > 0 else "-"
                    md += f"| {item['nazwa']} | {stan:.1f} | {min_stan:.0f} | {item['jednostka_miary']} | {status} | {cena_str} |\n"

                md += "\n"

            # Section: products to buy
            to_buy = [i for i in pantry_data if i['stan'] < i['minimum_ilosc']]
            if to_buy:
                md += "---\n\n"
                md += "## 🛒 Do kupienia\n\n"
                for item in sorted(to_buy, key=lambda x: x['stan']):
                    brakuje = item['minimum_ilosc'] - item['stan']
                    md += f"- [ ] **{item['nazwa']}** - brakuje ~{brakuje:.0f} {item['jednostka_miary']}\n"
                md += "\n"

            md += "---\n#spiżarnia #auto-generated\n"

        # Save to file
        target_path = ProjectConfig.OBSIDIAN_VAULT / "Zasoby" / "Spiżarnia.md"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        target_path.write_text(md, encoding='utf-8')
        print(f"Wyeksportowano stan spiżarni do {target_path.name}")
