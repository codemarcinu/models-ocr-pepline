import os
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

from sqlalchemy import create_engine, func, Column, Integer, String, Date, Numeric, ForeignKey, text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables from .env file
load_dotenv()

# Assuming ProjectConfig is available or can be mocked for now
# If ProjectConfig is from a specific file, we need to import it.
# For now, I'll define a dummy one if it's not globally available.
try:
    from config import ProjectConfig
except ImportError:
    class ProjectConfig:
        RECEIPT_DB_URL = os.getenv("RECEIPT_DB_URL", "postgresql://user:password@host:port/database")
        # Add other necessary configs if needed, or get from environment

Base = declarative_base()

# ==========================================
# 📊 NOWA STRUKTURA BAZY DANYCH (PostgreSQL)
# ==========================================

class Produkt(Base):
    __tablename__ = 'produkty'
    id = Column(Integer, primary_key=True)
    nazwa = Column(String(255), unique=True, nullable=False)
    kategoria = Column(String(100))
    jednostka_miary = Column(String(50))
    cena_zakupu = Column(Numeric(10, 2))
    dostawca = Column(String(255))
    minimum_ilosc = Column(Numeric(10, 2), default=1.0) # Nowe pole: próg alarmowy
    
    pozycje = relationship("PozycjaParagonu", back_populates="produkt")

class Paragon(Base):
    __tablename__ = 'paragony'
    id = Column(Integer, primary_key=True)
    data_zakupow = Column(Date, nullable=False)
    nazwa_sklepu = Column(String(255))
    suma_calkowita = Column(Numeric(10, 2))
    hash_identyfikacyjny = Column(String(64), unique=True) # Unikalny hash treści
    plik_zrodlowy = Column(String(512)) # Ścieżka do notatki
    
    pozycje = relationship("PozycjaParagonu", back_populates="paragon", cascade="all, delete-orphan")

class PozycjaParagonu(Base):
    __tablename__ = 'pozycje_paragonu'
    id = Column(Integer, primary_key=True)
    paragon_id = Column(Integer, ForeignKey('paragony.id'))
    produkt_id = Column(Integer, ForeignKey('produkty.id'))
    ilosc = Column(Numeric(10, 3))
    cena_jednostkowa = Column(Numeric(10, 2))
    rabat = Column(Numeric(10, 2))
    cena_po_rabacie = Column(Numeric(10, 2))
    cena_calkowita = Column(Numeric(10, 2))
    data_waznosci = Column(Date, nullable=True) # Nowe pole (opcjonalne)
    
    paragon = relationship("Paragon", back_populates="pozycje")
    produkt = relationship("Produkt", back_populates="pozycje")

class Posilek(Base):
    __tablename__ = 'posilki'
    id = Column(Integer, primary_key=True)
    produkt_id = Column(Integer, ForeignKey('produkty.id'))
    ilosc = Column(Numeric(10, 3))
    jednostka_miary = Column(String(50))
    data = Column(Date, default=func.current_date())


class ProductRepository:
    def __init__(self):
        db_url = ProjectConfig.RECEIPT_DB_URL
        if not db_url:
            raise ValueError("RECEIPT_DB_URL not configured.")
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _get_session(self):
        return self.Session()

    def get_product_by_name(self, name: str) -> Optional[Produkt]:
        session = self._get_session()
        try:
            return session.query(Produkt).filter(func.lower(Produkt.nazwa) == name.lower()).first()
        finally:
            session.close()

    def get_product_by_id(self, product_id: int) -> Optional[Produkt]:
        """Pobiera szczegóły produktu po ID."""
        session = self._get_session()
        try:
            return session.query(Produkt).filter_by(id=product_id).first()
        finally:
            session.close()

    def update_product(self, product_id: int, nazwa: Optional[str] = None, kategoria: Optional[str] = None,
                       jednostka: Optional[str] = None, minimum: Optional[float] = None) -> bool:
        """Aktualizuje dane produktu."""
        session = self._get_session()
        try:
            p = session.query(Produkt).filter_by(id=product_id).first()
            if not p:
                return False

            if nazwa is not None:
                p.nazwa = nazwa.strip().title()
            if kategoria is not None:
                p.kategoria = kategoria
            if jednostka is not None:
                p.jednostka_miary = jednostka
            if minimum is not None:
                p.minimum_ilosc = minimum

            session.commit()
            return True

        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_product(self, product_id: int, delete_transactions: bool = False) -> dict:
        """
        Usuwa produkt z bazy.
        Jeśli delete_transactions=True, usuwa też powiązane transakcje.
        Zwraca status operacji.
        """
        session = self._get_session()
        try:
            p = session.query(Produkt).filter_by(id=product_id).first()
            if not p:
                return {"success": False, "error": "Produkt nie znaleziony"}

            # Sprawdź powiązane transakcje
            purchases = session.query(PozycjaParagonu).filter_by(produkt_id=product_id).count()
            consumptions = session.query(Posilek).filter_by(produkt_id=product_id).count()

            if (purchases > 0 or consumptions > 0) and not delete_transactions:
                return {
                    "success": False,
                    "error": "Produkt ma powiązane transakcje",
                    "purchases": purchases,
                    "consumptions": consumptions,
                    "product_name": p.nazwa
                }

            # Usuń transakcje jeśli potwierdzono
            if delete_transactions:
                session.query(PozycjaParagonu).filter_by(produkt_id=product_id).delete()
                session.query(Posilek).filter_by(produkt_id=product_id).delete()

            product_name = p.nazwa
            session.delete(p)
            session.commit()

            return {
                "success": True,
                "product_name": product_name,
                "deleted_purchases": purchases if delete_transactions else 0,
                "deleted_consumptions": consumptions if delete_transactions else 0
            }

        except SQLAlchemyError as e:
            session.rollback()
            return {"success": False, "error": str(e)}
        finally:
            session.close()

    def adjust_stock(self, product_id: int, new_stock: float, reason: str = "Korekta inwentaryzacyjna") -> bool:
        """
        Koryguje stan produktu do podanej wartości.
        Tworzy wpis zużycia lub zakupu wyrównujący stan.
        """
        session = self._get_session()
        try:
            p = session.query(Produkt).filter_by(id=product_id).first()
            if not p:
                return False

            # Oblicz aktualny stan
            zakupiono = session.query(func.sum(PozycjaParagonu.ilosc)).filter(
                PozycjaParagonu.produkt_id == product_id
            ).scalar() or 0
            zuzyto = session.query(func.sum(Posilek.ilosc)).filter(
                Posilek.produkt_id == product_id
            ).scalar() or 0
            current_stock = float(zakupiono) - float(zuzyto)

            difference = new_stock - current_stock

            if abs(difference) < 0.001:
                return True  # Brak różnicy

            today = datetime.now().date()

            if difference > 0:
                # Trzeba dodać - tworzymy "paragon" korekty
                receipt_hash = f"ADJUST-{today}-{product_id}-{datetime.now().timestamp()}"
                paragon = Paragon(
                    data_zakupow=today,
                    nazwa_sklepu=reason,
                    suma_calkowita=0.0,
                    hash_identyfikacyjny=receipt_hash,
                    plik_zrodlowy="Stock Adjustment"
                )
                session.add(paragon)
                session.flush()

                pozycja = PozycjaParagonu(
                    paragon_id=paragon.id,
                    produkt_id=product_id,
                    ilosc=difference,
                    cena_jednostkowa=0.0,
                    rabat=0.0,
                    cena_po_rabacie=0.0,
                    cena_calkowita=0.0
                )
                session.add(pozycja)
            else:
                # Trzeba odjąć - tworzymy zużycie korekty
                posilek = Posilek(
                    produkt_id=product_id,
                    ilosc=abs(difference),
                    jednostka_miary=p.jednostka_miary,
                    data=today
                )
                session.add(posilek)

            session.commit()
            return True

        except SQLAlchemyError as e:
            session.rollback()
            print(f"Failed to adjust stock: {e}") # Using print for now, will integrate proper logging later
            return False
        finally:
            session.close()

    def merge_products(self, source_id: int, target_id: int) -> dict:
        """
        Łączy produkt źródłowy z docelowym.
        Przenosi wszystkie transakcje i usuwa produkt źródłowy.
        """
        session = self._get_session()
        try:
            source = session.query(Produkt).filter_by(id=source_id).first()
            target = session.query(Produkt).filter_by(id=target_id).first()

            if not source or not target:
                return {"success": False, "error": "Jeden z produktów nie istnieje"}

            if source_id == target_id:
                return {"success": False, "error": "Nie można połączyć produktu z samym sobą"}

            # Przenieś zakupy
            purchases_moved = session.query(PozycjaParagonu).filter_by(
                produkt_id=source_id
            ).update({"produkt_id": target_id})

            # Przenieś zużycia
            consumptions_moved = session.query(Posilek).filter_by(
                produkt_id=source_id
            ).update({"produkt_id": target_id})

            source_name = source.nazwa
            target_name = target.nazwa

            # Usuń produkt źródłowy
            session.delete(source)
            session.commit()

            return {
                "success": True,
                "source_name": source_name,
                "target_name": target_name,
                "purchases_moved": purchases_moved,
                "consumptions_moved": consumptions_moved
            }

        except SQLAlchemyError as e:
            session.rollback()
            return {"success": False, "error": str(e)}
        finally:
            session.close()

    def get_product_history(self, product_id: int) -> dict:
        """Pobiera historię transakcji produktu."""
        session = self._get_session()

        try:
            p = session.query(Produkt).filter_by(id=product_id).first()
            if not p:
                return {"error": "Produkt nie znaleziony"}

            # Zakupy
            purchases = session.query(PozycjaParagonu, Paragon).join(Paragon).filter(
                PozycjaParagonu.produkt_id == product_id
            ).order_by(Paragon.data_zakupow.desc()).all()

            purchases_list = [
                {
                    "id": poz.id,
                    "data": par.data_zakupow.strftime("%Y-%m-%d"),
                    "sklep": par.nazwa_sklepu,
                    "ilosc": float(poz.ilosc),
                    "cena": float(poz.cena_calkowita) if poz.cena_calkowita else 0.0
                }
                for poz, par in purchases
            ]

            # Zużycia
            consumptions = session.query(Posilek).filter_by(
                produkt_id=product_id
            ).order_by(Posilek.data.desc()).all()

            consumptions_list = [
                {
                    "id": c.id,
                    "data": c.data.strftime("%Y-%m-%d") if c.data else "-",
                    "ilosc": float(c.ilosc)
                }
                for c in consumptions
            ]

            return {
                "product_name": p.nazwa,
                "purchases": purchases_list,
                "consumptions": consumptions_list
            }

        except SQLAlchemyError as e:
            session.rollback()
            return {"error": str(e)}
        finally:
            session.close()

    def delete_consumption(self, consumption_id: int) -> dict:
        """Usuwa wpis zużycia (cofnięcie)."""
        session = self._get_session()

        try:
            posilek = session.query(Posilek).filter_by(id=consumption_id).first()
            if not posilek:
                return {"success": False, "error": "Zużycie nie znalezione"}

            # Pobierz nazwę produktu dla logu
            produkt = session.query(Produkt).filter_by(id=posilek.produkt_id).first()
            product_name = produkt.nazwa if produkt else "Unknown"
            ilosc = float(posilek.ilosc)

            session.delete(posilek)
            session.commit()

            return {
                "success": True,
                "product_name": product_name,
                "ilosc": ilosc
            }

        except SQLAlchemyError as e:
            session.rollback()
            return {"success": False, "error": str(e)}
        finally:
            session.close()






    def add_or_update_product(self, name: str, category: str, unit: str, price: float, supplier: str) -> Produkt:
        session = self._get_session()
        try:
            produkt = session.query(Produkt).filter(func.lower(Produkt.nazwa) == name.lower()).first()
            if not produkt:
                produkt = Produkt(
                    nazwa=name,
                    kategoria=category,
                    jednostka_miary=unit,
                    cena_zakupu=price,
                    dostawca=supplier
                )
                session.add(produkt)
            else:
                produkt.kategoria = category
                produkt.jednostka_miary = unit
                produkt.cena_zakupu = price
                produkt.dostawca = supplier
            session.commit()
            session.refresh(produkt)
            return produkt
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def save_transaction(self,
                         shop_name: str,
                         transaction_date: date,
                         total_sum: float,
                         receipt_hash: str,
                         source_file: str,
                         items_data: List[Dict]) -> Paragon:
        session = self._get_session()
        try:
            paragon = Paragon(
                data_zakupow=transaction_date,
                nazwa_sklepu=shop_name,
                suma_calkowita=total_sum,
                hash_identyfikacyjny=receipt_hash,
                plik_zrodlowy=source_file
            )
            session.add(paragon)
            session.flush() # Flush to get paragon.id

            for item_data in items_data:
                # Ensure product exists or create it
                produkt = self.add_or_update_product(
                    name=item_data['nazwa'],
                    category=item_data.get('kategoria', 'Inne'),
                    unit=item_data.get('jednostka', 'szt'),
                    price=item_data.get('cena_jedn', 0.0),
                    supplier=shop_name # supplier is shop for now
                )

                exp_date = None
                if item_data.get('data_waznosci'):
                    try:
                        exp_date = datetime.strptime(item_data['data_waznosci'], "%Y-%m-%d").date()
                    except ValueError:
                        pass # Ignore if date format is incorrect

                poz = PozycjaParagonu(
                    paragon_id=paragon.id,
                    produkt_id=produkt.id,
                    ilosc=item_data['ilosc'],
                    cena_jednostkowa=item_data.get('cena_jedn', 0.0),
                    rabat=item_data.get('rabat', 0.0),
                    cena_po_rabacie=item_data.get('cena_jedn', 0.0) - (item_data.get('rabat', 0.0)/item_data['ilosc'] if item_data['ilosc'] > 0 else 0),
                    cena_calkowita=item_data.get('suma', 0.0),
                    data_waznosci=exp_date
                )
                session.add(poz)
            session.commit()
            session.refresh(paragon)
            return paragon
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def add_consumption(self, product_name: str, quantity: float, unit: str, consumption_date: Optional[date] = None) -> Optional[Posilek]:
        session = self._get_session()
        try:
            produkt = session.query(Produkt).filter(func.lower(Produkt.nazwa) == product_name.lower()).first()
            if not produkt:
                # If product doesn't exist, we can't log consumption for it.
                # In a real scenario, this might trigger a creation or alert.
                return None
            
            consumption = Posilek(
                produkt_id=produkt.id,
                ilosc=quantity,
                jednostka_miary=unit, # Use the unit from consumption, or fallback to product's unit
                data=consumption_date or datetime.now().date()
            )
            session.add(consumption)
            session.commit()
            session.refresh(consumption)
            return consumption
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def get_pantry_state(self) -> List[Dict]:
        """
        Retrieves the current state of all products in the pantry, including calculated stock.
        Returns pure data, not Markdown.
        """
        session = self._get_session()
        try:
            produkty = session.query(Produkt).order_by(Produkt.kategoria, Produkt.nazwa).all()
            pantry_data = []
            for p in produkty:
                zakupiono = session.query(func.sum(PozycjaParagonu.ilosc)).filter(
                    PozycjaParagonu.produkt_id == p.id
                ).scalar() or 0
                zuzyto = session.query(func.sum(Posilek.ilosc)).filter(
                    Posilek.produkt_id == p.id
                ).scalar() or 0
                stan = float(zakupiono) - float(zuzyto)

                pantry_data.append({
                    'id': p.id,
                    'kategoria': p.kategoria or 'Inne',
                    'nazwa': p.nazwa,
                    'stan': round(stan, 2),
                    'minimum_ilosc': float(p.minimum_ilosc) if p.minimum_ilosc else 1.0,
                    'jednostka_miary': p.jednostka_miary or 'szt',
                    'cena_zakupu': float(p.cena_zakupu) if p.cena_zakupu else 0.0,
                    'dostawca': p.dostawca or '-'
                })
            return pantry_data
        finally:
            session.close()

    def get_product_stock_and_min_level(self, product_name: str) -> Optional[Tuple[float, float]]:
        """
        Returns the current stock and minimum level for a given product name.
        """
        session = self._get_session()
        try:
            produkt = session.query(Produkt).filter(func.lower(Produkt.nazwa) == product_name.lower()).first()
            if not produkt:
                return None

            zakupiono = session.query(func.sum(PozycjaParagonu.ilosc)).filter(
                PozycjaParagonu.produkt_id == produkt.id
            ).scalar() or 0
            zuzyto = session.query(func.sum(Posilek.ilosc)).filter(
                Posilek.produkt_id == produkt.id
            ).scalar() or 0
            stan = float(zakupiono) - float(zuzyto)
            min_ilosc = float(produkt.minimum_ilosc) if produkt.minimum_ilosc else 1.0
            return stan, min_ilosc
        finally:
            session.close()
            
    def get_all_products_details(self) -> List[Produkt]:
        """
        Returns all product objects.
        """
        session = self._get_session()
        try:
            return session.query(Produkt).all()
        finally:
            session.close()

    def get_unprocessed_receipt_hashes(self, hashes: List[str]) -> List[str]:
        """
        Checks a list of receipt hashes and returns those not yet in the database.
        """
        session = self._get_session()
        try:
            existing_hashes = {r.hash_identyfikacyjny for r in session.query(Paragon.hash_identyfikacyjny).filter(Paragon.hash_identyfikacyjny.in_(hashes)).all()}
            return [h for h in hashes if h not in existing_hashes]
        finally:
            session.close()

    def add_consumptions_bulk(self, consumption_list: List[Tuple[int, float, str]], consumption_date: Optional[date] = None) -> bool:
        """
        Records the consumption of multiple products in a single transaction.
        consumption_list: list of tuples (product_id, quantity, unit)
        """
        session = self._get_session()
        try:
            for product_id, quantity, unit in consumption_list:
                if quantity <= 0:
                    continue
                consumption = Posilek(
                    produkt_id=product_id,
                    ilosc=quantity,
                    jednostka_miary=unit,
                    data=consumption_date or datetime.now().date()
                )
                session.add(consumption)
            session.commit()
            return True
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_purchase(self, purchase_id: int) -> dict:
        """Usuwa wpis zakupu (pozycja paragonu)."""
        session = self._get_session()
        try:
            pozycja = session.query(PozycjaParagonu).filter_by(id=purchase_id).first()
            if not pozycja:
                return {"success": False, "error": "Zakup nie znaleziony"}

            produkt = session.query(Produkt).filter_by(id=pozycja.produkt_id).first()
            product_name = produkt.nazwa if produkt else "Unknown"
            ilosc = float(pozycja.ilosc)

            session.delete(pozycja)
            session.commit()

            return {
                "success": True,
                "product_name": product_name,
                "ilosc": ilosc
            }

        except SQLAlchemyError as e:
            session.rollback()
            return {"success": False, "error": str(e)}
        finally:
            session.close()

    def get_product_stock_details_by_id(self, product_id: int) -> Optional[Dict]:
        """
        Returns stock details for a given product ID.
        """
        session = self._get_session()
        try:
            p = session.query(Produkt).filter_by(id=product_id).first()
            if not p:
                return None

            zakupiono = session.query(func.sum(PozycjaParagonu.ilosc)).filter(PozycjaParagonu.produkt_id == p.id).scalar() or 0
            zuzyto = session.query(func.sum(Posilek.ilosc)).filter(Posilek.produkt_id == p.id).scalar() or 0
            stan = float(zakupiono) - float(zuzyto)

            return {
                "id": p.id,
                "nazwa": p.nazwa,
                "kategoria": p.kategoria,
                "jednostka": p.jednostka_miary,
                "minimum": float(p.minimum_ilosc) if p.minimum_ilosc else 1.0,
                "cena": float(p.cena_zakupu) if p.cena_zakupu else 0.0,
                "dostawca": p.dostawca,
                "stan": round(stan, 2),
                "zakupiono": float(zakupiono),
                "zuzyto": float(zuzyto)
            }
        finally:
            session.close()

    def get_expiring_products(self, days=7) -> List[Dict]:
        """Szuka produktów, których data ważności kończy się wkrótce."""
        session = self._get_session()

        today = datetime.now().date()
        deadline = today + timedelta(days=days)

        results = session.query(PozycjaParagonu, Produkt).join(Produkt).filter(
            PozycjaParagonu.data_waznosci <= deadline,
            PozycjaParagonu.data_waznosci >= today
        ).all()

        return [
            {
                "Produkt": p.nazwa,
                "Data": poz.data_waznosci.strftime("%Y-%m-%d"),
                "Dni": (poz.data_waznosci - today).days
            } for poz, p in results
        ]
