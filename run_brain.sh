#!/bin/bash

# Ustalenie katalogu skryptu
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Kolory
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}   🧠 OBSIDIAN AI SECOND BRAIN V2.0      ${NC}"
echo -e "${BLUE}==========================================${NC}"

# 1. Sprawdzenie venv
if [ -d "venv" ]; then
    echo -e "${GREEN}[1/6]${NC} Aktywacja wirtualnego środowiska..."
    source venv/bin/activate
else
    echo -e "${YELLOW}[1/6]${NC} Nie znaleziono venv. Tworzenie nowego środowiska..."
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${GREEN}      Gotowe.${NC}"
fi

# Dodanie bieżącego katalogu do PYTHONPATH
export PYTHONPATH=$PWD

# 2. Sprawdzenie narzędzi systemowych (FFmpeg)
echo -e "${BLUE}[2/6]${NC} Sprawdzanie narzędzi systemowych..."
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ BŁĄD: Nie znaleziono ffmpeg!${NC}"
    echo -e "Jest on wymagany do przetwarzania audio/wideo."
    echo -e "Zainstaluj go komendą: ${YELLOW}sudo apt install -y ffmpeg${NC}"
    exit 1
else
    echo -e "${GREEN}      FFmpeg jest zainstalowany.${NC}"
fi

# 3. Aktualizacja zależności
echo -e "${BLUE}[3/6]${NC} Weryfikacja bibliotek..."
pip install -r requirements.txt | grep -v "already satisfied" || true
echo -e "${GREEN}      Biblioteki sprawdzone.${NC}"

# 4. Sprawdzenie konfiguracji i czyszczenie
echo -e "${BLUE}[4/6]${NC} Przygotowanie środowiska (czyszczenie temp)..."
mkdir -p obsidian_db/_INBOX
mkdir -p temp_processing
mkdir -p logs
# Usuwanie plików starszych niż 24h z temp
find temp_processing -type f -mmin +1440 -delete 2>/dev/null || true

# Pobranie AI_PROVIDER z .env
AI_PROVIDER=$(grep "^AI_PROVIDER=" .env | cut -d'=' -f2 | tr -d '"' | tr -d "'")
[ -z "$AI_PROVIDER" ] && AI_PROVIDER="gemini"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}      UWAGA: Brak pliku .env. Uruchamiam z domyślnymi ustawieniami.${NC}"
else
    echo -e "${GREEN}      Plik .env wczytany (Dostawca AI: $AI_PROVIDER).${NC}"
fi

# 5. Sprawdzenie Modelu AI
if [ "$AI_PROVIDER" == "local" ]; then
    echo -e "${BLUE}[5/7]${NC} Sprawdzanie lokalnego modelu AI (Ollama)..."
    python3 check_ollama.py
    echo -e "${GREEN}      Gotowe.${NC}"
else
    echo -e "${GREEN}[5/7]${NC} Tryb Cloud-Native ($AI_PROVIDER). Pomijam sprawdzanie Ollama.${NC}"
fi

# 6. Sprawdzenie Google Calendar Credentials
echo -e "${BLUE}[6/7]${NC} Sprawdzanie dostępu do Google Calendar..."
if [ ! -f "credentials.json" ]; then
    echo -e "${YELLOW}      INFO: Brak credentials.json. Integracja z kalendarzem będzie nieaktywna.${NC}"
else
    echo -e "${GREEN}      Credentials znalezione.${NC}"
fi

# --- URUCHOMIENIE SYSTEMÓW TŁA ---
echo -e "${BLUE}[+]${NC} Uruchamianie Systemów Tła..."

# Funkcja czyszcząca - zabija procesy w tle przy wyjściu (Ctrl+C)
cleanup() {
    echo -e "\n${YELLOW}Zamykanie systemu...${NC}"
    [ ! -z "$GUARD_PID" ] && kill $GUARD_PID
    [ ! -z "$SERVER_PID" ] && kill $SERVER_PID
    [ ! -z "$CALENDAR_PID" ] && kill $CALENDAR_PID
    [ ! -z "$DRIVE_PID" ] && kill $DRIVE_PID
    [ ! -z "$FIXER_PID" ] && kill $FIXER_PID
    [ ! -z "$LOG_PID" ] && kill $LOG_PID
    exit
}

# Rejestracja sygnału wyjścia
trap cleanup SIGINT SIGTERM

# Start Strażnika w tle
python3 -u brain_guard.py > logs/brain_guard_runtime.log 2>&1 &
GUARD_PID=$!
sleep 2
if ! kill -0 $GUARD_PID 2>/dev/null; then
    echo -e "${RED}❌ BŁĄD: BrainGuard (Strażnik) natychmiast zakończył działanie!${NC}"
    echo -e "${YELLOW}Sprawdzam logi (logs/brain_guard_runtime.log):${NC}"
    tail -n 10 logs/brain_guard_runtime.log
    cleanup
    exit 1
fi
echo -e "${GREEN}      Strażnik (BrainGuard) działa (PID: $GUARD_PID).${NC}"

# Start Brain Bridge API
PORT=8000
echo -e "${BLUE}[INFO]${NC} Sprawdzanie portu $PORT..."
if lsof -ti:$PORT >/dev/null; then
    echo -e "${YELLOW}Port $PORT jest zajęty. Zabijanie procesu...${NC}"
    lsof -ti:$PORT | xargs kill -9
    sleep 1
fi

python3 -u server.py > logs/server_runtime.log 2>&1 &
SERVER_PID=$!
sleep 2
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo -e "${RED}❌ BŁĄD: Brain Bridge API (server.py) natychmiast zakończył działanie!${NC}"
    echo -e "${YELLOW}Sprawdzam logi (logs/server_runtime.log):${NC}"
    tail -n 10 logs/server_runtime.log
    cleanup
    exit 1
fi
echo -e "${GREEN}      Brain Bridge API działa (PID: $SERVER_PID).${NC}"

# Start Calendar Bridge (tylko jeśli są credentials)
if [ -f "credentials.json" ]; then
    python3 -u adapters/google/calendar_adapter.py --service > logs/calendar_runtime.log 2>&1 &
    CALENDAR_PID=$!
    sleep 2
    if ! kill -0 $CALENDAR_PID 2>/dev/null; then
        echo -e "${RED}❌ BŁĄD: Calendar Bridge natychmiast zakończył działanie!${NC}"
        echo -e "${YELLOW}Sprawdzam logi (logs/calendar_runtime.log):${NC}"
        tail -n 10 logs/calendar_runtime.log
        cleanup
        exit 1
    fi
    echo -e "${GREEN}      Calendar Bridge działa (PID: $CALENDAR_PID).${NC}"

        # Start Drive Bridge

        python3 -u adapters/google/drive_adapter.py --service > logs/drive_runtime.log 2>&1 &
        DRIVE_PID=$!
        sleep 2
        if ! kill -0 $DRIVE_PID 2>/dev/null; then
            echo -e "${RED}❌ BŁĄD: Drive Bridge natychmiast zakończył działanie!${NC}"
            echo -e "${YELLOW}Sprawdzam logi (logs/drive_runtime.log):${NC}"
            tail -n 10 logs/drive_runtime.log
            cleanup
            exit 1
        fi
        echo -e "${GREEN}      Drive Bridge działa (PID: $DRIVE_PID).${NC}"

    

        # Start Daily Note Fixer

        python3 -u ensure_calendar_section.py --service > logs/calendar_fixer_runtime.log 2>&1 &
        FIXER_PID=$!
        sleep 2
        if ! kill -0 $FIXER_PID 2>/dev/null; then
            echo -e "${RED}❌ BŁĄD: Daily Note Fixer natychmiast zakończył działanie!${NC}"
            echo -e "${YELLOW}Sprawdzam logi (logs/calendar_fixer_runtime.log):${NC}"
            tail -n 10 logs/calendar_fixer_runtime.log
            cleanup
            exit 1
        fi
        echo -e "${GREEN}      Daily Note Fixer działa (PID: $FIXER_PID).${NC}"

    fi

    

    # Start Log Dashboard
    LOG_PORT=8001
    echo -e "${BLUE}[INFO]${NC} Sprawdzanie portu $LOG_PORT..."
    if lsof -ti:$LOG_PORT >/dev/null; then
        echo -e "${YELLOW}Port $LOG_PORT jest zajęty. Zabijanie procesu...${NC}"
        lsof -ti:$LOG_PORT | xargs kill -9
        sleep 1
    fi

    python3 -u core/services/log_server.py > logs/log_server_runtime.log 2>&1 &
    LOG_PID=$!
    sleep 2
    if ! kill -0 $LOG_PID 2>/dev/null; then
        echo -e "${RED}❌ BŁĄD: Log Dashboard natychmiast zakończył działanie!${NC}"
        echo -e "${YELLOW}Sprawdzam logi (logs/log_server_runtime.log):${NC}"
        tail -n 10 logs/log_server_runtime.log
        cleanup
        exit 1
    fi
    echo -e "${GREEN}      Log Dashboard działa (PID: $LOG_PID) -> http://localhost:8001${NC}"

    # 7. Start aplikacji
    echo -e "${BLUE}[7/7]${NC} Uruchamianie interfejsu Brain CLI..."
    echo -e "${YELLOW}----------------------------------------------------------${NC}"
    echo -e "${GREEN}  🧠 Witaj w Twoim Drugim Mózgu! ${NC}"
    echo -e "${YELLOW}----------------------------------------------------------${NC}"
    
    python3 brain.py

    