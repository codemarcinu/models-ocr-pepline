# Raport Funkcjonalności Projektu: Obsidian AI Second Brain / BrainGuard

## 1. Wstęp
Ten dokument stanowi szczegółowy raport dotyczący funkcjonalności systemu "Obsidian AI Second Brain" (znanego również jako BrainGuard/IronBrain). Projekt ten jest zaawansowanym, osobistym asystentem cyfrowym, zaprojektowanym z myślą o prywatności, automatyzacji codziennych zadań oraz zarządzaniu wiedzą.

W przeciwieństwie do popularnych usług chmurowych (jak ChatGPT), ten system działa w dużej mierze lokalnie na Twoim komputerze, co zapewnia pełną kontrolę nad danymi.

## 2. Główna Idea: "Drugi Mózg"
System działa jako "cyfrowa pamięć zewnętrzna". Jego celem jest odciążenie użytkownika od konieczności pamiętania o drobiazgach, organizowanie informacji oraz pomoc w nauce i pracy. System łączy w sobie cechy:
- **Sekretarza:** Zarządza kalendarzem i mailami.
- **Bibliotekarza:** Organizuje notatki i dokumenty.
- **Zarządcy domu:** Pilnuje zapasów i wydatków.

## 3. Szczegółowy Opis Funkcji

### A. Inteligentna Spiżarnia i Finanse
System rewolucjonizuje sposób zarządzania domowym budżetem i zapasami:
- **Skanowanie Paragonów:** Wystarczy zrobić zdjęcie lub wrzucić plik PDF z paragonem. System (używając technologii OCR i AI) automatycznie odczytuje listę zakupów, kategoryzuje produkty i zapisuje wydatki.
- **Wirtualna Spiżarnia:** Na podstawie paragonów system aktualizuje stany magazynowe w domu ("wiem, że mam 2 masła").
- **Komendy głosowe/tekstowe:** Można ręcznie dodać produkt ("Kupiłem truskawki") lub go "zużyć" ("Zjadłem jogurt"), aby utrzymać porządek.

### B. Przetwarzanie Wiedzy (YouTube i Audio)
Narzędzie to jest potężnym wsparciem w nauce i konsumpcji treści:
- **Streszczanie Filmów:** Użytkownik wkleja link do filmu na YouTube, a system "ogląda" go, tworzy transkrypcję i generuje zwięzłą notatkę z najważniejszymi informacjami.
- **Notatki Głosowe:** Nagrania z dyktafonu są automatycznie zamieniane na tekst i formatowane jako czytelne notatki.
- **Obsługa PDF:** Długie dokumenty i artykuły są analizowane i streszczane, co pozwala zaoszczędzić czas na czytaniu.

### C. "Strażnik" (BrainGuard) - Automatyzacja w Tle
Jest to niewidoczny proces działający w tle, który obserwuje wyznaczony folder (Inbox):
- **Automatyczne Sortowanie:** Cokolwiek wrzucisz do folderu (Faktura, PDF, zdjęcie, notatka tekstowa), system to rozpozna.
- **Inteligentne Przetwarzanie:** Faktury trafiają do finansów, artykuły do bazy wiedzy, a zdjęcia są opisywane. Dzieje się to bez udziału użytkownika.
- **Wielozadaniowość:** System potrafi przetwarzać wiele plików jednocześnie.

### D. Asystent Email (Gmail Integration)
System pomaga utrzymać porządek w skrzynce mailowej (Inbox Zero):
- **Inteligentna Segregacja:** AI analizuje przychodzące maile, oddzielając ważne wiadomości (np. faktury) od newsletterów i spamu.
- **Automatyzacja Zadań:** Na podstawie treści maila system może utworzyć zadanie w kalendarzu lub notatkę w Obsidianie.
- **Bezpieczeństwo:** Użytkownik ma podgląd działań (tryb "Dry Run") przed ich faktycznym wykonaniem.

### E. Czat z Własną Wiedzą (RAG)
Najważniejsza funkcja wyróżniająca ten system:
- **Kontekst:** Możesz rozmawiać z asystentem (modelem "Bielik"), który **zna Twoje notatki**.
- **Pytania:** Możesz zapytać: "Kiedy ostatnio byłem u lekarza?" lub "Co pisałem o projekcie X rok temu?", a system znajdzie odpowiedź w Twoich prywatnych plikach, a nie w internecie.

## 4. Aspekty Techniczne (W Języku Prostym)

- **Prywatność (Local First):** Większość obliczeń (analiza tekstu, rozmowa z AI) odbywa się na Twoim komputerze, wykorzystując Twoją kartę graficzną. Dane nie wyciekają do chmury.
- **Inteligentny Router (SmartRouter):** System jest oszczędny. Do prostych zadań (np. odczytanie kwoty z paragonu) używa darmowych, lokalnych modeli. Tylko do bardzo trudnych zadań (rozpoznanie skomplikowanego obrazu) łączy się z płatną chmurą (Google Gemini), pilnując ustalonego budżetu.
- **Integracja:** System łączy się z Twoim Kalendarzem Google, Dyskiem Google i listą zadań, tworząc jeden spójny ekosystem.

## 5. Podsumowanie Korzyści
Dla użytkownika system ten oznacza:
1.  **Oszczędność Czasu:** Koniec z ręcznym wpisywaniem wydatków czy przepisywaniem notatek.
2.  **Porządek:** Wszystkie dokumenty i pliki są automatycznie nazywane i katalogowane.
3.  **Spokój Ducha:** System pamięta o terminach i "ma wszystko pod kontrolą".
4.  **Łatwy Dostęp do Wiedzy:** Błyskawiczne wyszukiwanie informacji we własnych zbiorach.

Projekt jest gotowym, kompleksowym rozwiązaniem klasy "Second Brain", które wykracza poza zwykłe notowanie, stając się aktywnym uczestnikiem życia cyfrowego użytkownika.
