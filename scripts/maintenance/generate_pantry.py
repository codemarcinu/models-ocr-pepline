#!/usr/bin/env python3
"""
Prosty skrypt do wygenerowania pliku Spiżarnia.md bez wszystkich zależności projektu.
"""
import os
import sys
from pathlib import Path
from datetime import datetime

from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker

# Odczyt RECEIPT_DB_URL z .env
env_file = Path(__file__).parent / ".env"
DB_URL = None
VAULT_PATH = None

if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith("RECEIPT_DB_URL="):
            DB_URL = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("OBSIDIAN_VAULT="):
            VAULT_PATH = line.split("=", 1)[1].strip().strip('"').strip("'")

if not DB_URL:
    print("❌ Brak RECEIPT_DB_URL w .env")
    sys.exit(1)

if not VAULT_PATH:
    VAULT_PATH = "/mnt/c/Users/marci/Documents/Obsidian Vault"

# Emoji dla kategorii
CAT_EMOJIS = {
    "NABIAŁ": "🥛", "MIĘSO_WĘDLINY": "🥩", "RYBY": "🐟",
    "WARZYWA_OWOCE": "🍎", "SPIŻARNIA": "🍝", "PIECZYWO": "🍞",
    "NAPOJE": "🧃", "DANIA_GOTOWE": "🍕", "PRZETWORY": "🥫",
    "ŚNIADANIE": "🥣", "SUCHE": "🍝", "DODATKI": "🧂",
    "PRZEKĄSKI": "🍿", "CHEMIA_HIGIENA": "🧼",
    "ELEKTRONIKA_DOM": "💡", "UŻYWKI": "🚬", "INNE": "📦"
}

def generate_spizarnia():
    engine = create_engine(DB_URL)

    with engine.connect() as conn:
        # Pobierz produkty
        products_result = conn.execute(text("""
            SELECT id, nazwa, kategoria, jednostka_miary, cena_zakupu, dostawca, minimum_ilosc
            FROM produkty
            ORDER BY kategoria, nazwa
        """))
        products = products_result.fetchall()

        if not products:
            print("❌ Brak produktów w bazie")
            return

        # Oblicz stan dla każdego produktu
        pantry_data = []
        for p in products:
            prod_id, nazwa, kategoria, jednostka, cena, dostawca, minimum = p

            # Zakupiono
            zakup_result = conn.execute(text("""
                SELECT COALESCE(SUM(ilosc), 0) FROM pozycje_paragonu WHERE produkt_id = :id
            """), {"id": prod_id})
            zakupiono = float(zakup_result.scalar() or 0)

            # Zużyto
            zuzycie_result = conn.execute(text("""
                SELECT COALESCE(SUM(ilosc), 0) FROM posilki WHERE produkt_id = :id
            """), {"id": prod_id})
            zuzyto = float(zuzycie_result.scalar() or 0)

            stan = zakupiono - zuzyto

            pantry_data.append({
                'kategoria': kategoria or 'INNE',
                'nazwa': nazwa,
                'stan': round(stan, 2),
                'min': float(minimum) if minimum else 1.0,
                'jednostka': jednostka or 'szt',
                'cena': float(cena) if cena else 0.0,
                'dostawca': dostawca or '-'
            })

    # Grupowanie po kategorii
    by_category = {}
    for item in pantry_data:
        cat = item['kategoria']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)

    # Statystyki
    total_items = len(pantry_data)
    in_stock = sum(1 for i in pantry_data if i['stan'] > 0)
    low_stock = sum(1 for i in pantry_data if 0 < i['stan'] < i['min'])
    out_of_stock = sum(1 for i in pantry_data if i['stan'] <= 0)

    # Budowanie MD
    md = f"# 📦 Stan Spiżarni\n"
    md += f"> Ostatnia aktualizacja: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    md += f"**Podsumowanie:** ✅ {in_stock} w magazynie | ⚠️ {low_stock} niski stan | ❌ {out_of_stock} brak\n\n"
    md += "---\n\n"

    # Tabela główna pogrupowana po kategorii
    for cat in sorted(by_category.keys()):
        items = by_category[cat]
        emoji = CAT_EMOJIS.get(cat, "📦")

        md += f"## {emoji} {cat}\n\n"
        md += "| Produkt | Stan | Min | Jednostka | Status | Ost. cena |\n"
        md += "|---------|-----:|----:|-----------|:------:|----------:|\n"

        for item in sorted(items, key=lambda x: x['nazwa']):
            stan = item['stan']
            min_stan = item['min']

            if stan <= 0:
                status = "❌"
            elif stan < min_stan:
                status = "⚠️"
            else:
                status = "✅"

            cena_str = f"{item['cena']:.2f} zł" if item['cena'] > 0 else "-"
            md += f"| {item['nazwa']} | {stan:.1f} | {min_stan:.0f} | {item['jednostka']} | {status} | {cena_str} |\n"

        md += "\n"

    # Sekcja: produkty do kupienia
    to_buy = [i for i in pantry_data if i['stan'] < i['min']]
    if to_buy:
        md += "---\n\n"
        md += "## 🛒 Do kupienia\n\n"
        for item in sorted(to_buy, key=lambda x: x['stan']):
            brakuje = item['min'] - item['stan']
            md += f"- [ ] **{item['nazwa']}** - brakuje ~{brakuje:.0f} {item['jednostka']}\n"
        md += "\n"

    md += "---\n#spiżarnia #auto-generated\n"

    # Zapis do pliku
    target_path = Path(VAULT_PATH) / "Zasoby" / "Spiżarnia.md"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    target_path.write_text(md, encoding='utf-8')
    print(f"✅ Wyeksportowano {total_items} produktów do {target_path}")
    print(f"   📊 W magazynie: {in_stock} | Niski stan: {low_stock} | Brak: {out_of_stock}")

if __name__ == "__main__":
    generate_spizarnia()
