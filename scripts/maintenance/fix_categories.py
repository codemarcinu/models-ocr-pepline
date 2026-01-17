#!/usr/bin/env python3
"""
Jednorazowy skrypt do naprawy kategorii w bazie danych PostgreSQL.
Uruchom: python fix_categories.py
"""
import os
import sys

# Dodanie ścieżki projektu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

try:
    from config import ProjectConfig
    DB_URL = ProjectConfig.RECEIPT_DB_URL
except ImportError:
    # Fallback - próba odczytania bezpośrednio z .env
    from pathlib import Path
    env_file = Path(__file__).parent / ".env"
    DB_URL = None
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("RECEIPT_DB_URL="):
                DB_URL = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

# Mapowanie błędnych kategorii na poprawne
CATEGORY_FIXES = {
    # Błędne kategorie "A", "B", "C" na prawidłowe
    "A": "NAPOJE",           # Piwo było oznaczone jako "A"
    "B": "SPIŻARNIA",        # Sosy były oznaczone jako "B"
    "C": "INNE",             # Różne produkty były oznaczone jako "C"

    # Dodatkowe normalizacje
    "Dodatki": "SPIŻARNIA",
    "Ryby": "RYBY",
    "Nabiał": "NABIAŁ",
    "Przekąski": "PRZEKĄSKI",
    "Przetwory": "PRZETWORY",
    "Śniadanie": "ŚNIADANIE",
    "Suche": "SPIŻARNIA",
}

# Mapowanie konkretnych produktów na kategorie (bardziej precyzyjne)
PRODUCT_CATEGORY_MAP = {
    # Nabiał
    "Mleko": "NABIAŁ",
    "Mleko 2%": "NABIAŁ",
    "Mleko UHT 2%": "NABIAŁ",
    "Jogurt Skyr naturalny": "NABIAŁ",
    "Jajka": "NABIAŁ",

    # Pieczywo
    "Bułka Poznańska z makiem": "PIECZYWO",

    # Mięso
    "Szarpana wieprzowina": "MIĘSO_WĘDLINY",
    "Parówki": "MIĘSO_WĘDLINY",

    # Warzywa/Owoce
    "Banany": "WARZYWA_OWOCE",
    "Banan Luz": "WARZYWA_OWOCE",

    # Napoje
    "Piwo Zatecky": "NAPOJE",
    "Kawa rozpuszczalna": "NAPOJE",

    # Spiżarnia
    "Sos American Made 38": "SPIŻARNIA",
    "Makaron spaghetti": "SPIŻARNIA",
    "Płatki kukurydziane": "SPIŻARNIA",
    "Buraczki zasmażane": "SPIŻARNIA",
    "Keczup pikantny Pudliszki": "SPIŻARNIA",
    "Majonez Kielecki": "SPIŻARNIA",

    # Przekąski
    "Ciastka chałwa": "PRZEKĄSKI",
    "Ciastka Chał": "PRZEKĄSKI",
    "Ciastka orzechowe": "PRZEKĄSKI",
    "Ciastka Orze": "PRZEKĄSKI",
    "Kabanosy Tarczyński chilli": "PRZEKĄSKI",

    # Ryby
    "Filety z makreli w sosie pomidorowym": "RYBY",

    # Dania gotowe
    "Naleśniki jabłko": "DANIA_GOTOWE",
    "Naleśniki ser": "DANIA_GOTOWE",
}


def fix_categories():
    """Naprawia kategorie w bazie danych."""
    db_url = DB_URL
    if not db_url:
        print("❌ Brak RECEIPT_DB_URL w konfiguracji!")
        sys.exit(1)

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    print("🔍 Analizowanie produktów w bazie...")

    # Pobranie wszystkich produktów
    result = session.execute(text("SELECT id, nazwa, kategoria FROM produkty"))
    products = result.fetchall()

    updates = []

    for prod_id, nazwa, kategoria in products:
        new_cat = None

        # 1. Sprawdź czy produkt ma dedykowane mapowanie
        if nazwa in PRODUCT_CATEGORY_MAP:
            new_cat = PRODUCT_CATEGORY_MAP[nazwa]
        # 2. Sprawdź czy kategoria wymaga naprawy
        elif kategoria in CATEGORY_FIXES:
            new_cat = CATEGORY_FIXES[kategoria]

        if new_cat and new_cat != kategoria:
            updates.append((prod_id, nazwa, kategoria, new_cat))

    if not updates:
        print("✅ Wszystkie kategorie są poprawne. Brak zmian do wprowadzenia.")
        session.close()
        return

    print(f"\n📋 Znaleziono {len(updates)} produktów do naprawy:\n")
    print(f"{'ID':<5} {'Produkt':<35} {'Stara kat.':<15} → {'Nowa kat.':<15}")
    print("-" * 75)

    for prod_id, nazwa, old_cat, new_cat in updates:
        print(f"{prod_id:<5} {nazwa[:33]:<35} {old_cat:<15} → {new_cat:<15}")

    print("\n" + "-" * 75)

    # Automatyczne zatwierdzenie jeśli podano --yes
    if "--yes" in sys.argv or "-y" in sys.argv:
        confirm = 't'
        print("\n✅ Automatyczne zatwierdzenie (--yes)")
    else:
        try:
            confirm = input("\n❓ Czy chcesz wprowadzić te zmiany? (t/n): ").strip().lower()
        except EOFError:
            confirm = 't'  # W trybie nieinteraktywnym zatwierdź

    if confirm != 't':
        print("❌ Anulowano.")
        session.close()
        return

    # Wykonanie aktualizacji
    print("\n🔄 Aktualizowanie kategorii...")

    for prod_id, nazwa, old_cat, new_cat in updates:
        session.execute(
            text("UPDATE produkty SET kategoria = :new_cat WHERE id = :id"),
            {"new_cat": new_cat, "id": prod_id}
        )

    session.commit()
    print(f"✅ Zaktualizowano {len(updates)} produktów!")

    session.close()


if __name__ == "__main__":
    fix_categories()
