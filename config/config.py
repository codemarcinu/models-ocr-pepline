import os
import logging
import logging.handlers
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Setup Logging
logger = logging.getLogger("BrainOS")
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Timed Rotating File Handler
# Rotates logs every midnight, keeps last 7 days
file_handler = logging.handlers.TimedRotatingFileHandler(
    filename="logs/system.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

class Settings:
    """Application settings."""
    def __init__(self):
        self.load_config()

    def load_config(self):
        pass

class ProjectConfig:
    """Static configuration class."""
    
    # Paths
    BASE_DIR = Path(os.getcwd())
    OBSIDIAN_VAULT = Path(os.getenv("OBSIDIAN_VAULT_PATH", "/home/marcin/obsidian/obsidian_vault_test"))
    
    # AI Providers
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Models
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "SpeakLeash/bielik-11b-v2.3-instruct:Q5_K_M")
    OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", "gemma3:4b")
    
    # Obsidian PKM Models
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
    OLLAMA_GENERATION_MODEL = os.getenv("OLLAMA_GENERATION_MODEL", "SpeakLeash/bielik-11b-v2.3-instruct:Q5_K_M")
    OLLAMA_RECEIPT_MODEL = os.getenv("OLLAMA_RECEIPT_MODEL", "deepseek-r1:latest")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
    
    # AI Provider
    AI_PROVIDER = os.getenv("AI_PROVIDER", "gemini")
    RECEIPT_AI_PROVIDER = os.getenv("RECEIPT_AI_PROVIDER", "local")
    
    # Cloud Credentials
    GOOGLE_APPLICATION_CREDENTIALS = Path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/google_credentials.json"))
    
    # Configs
    RESOURCES_DIR = BASE_DIR / "config"
    TAGS_FILE = RESOURCES_DIR / "tags.yaml"
    PROMPTS_FILE = RESOURCES_DIR / "prompts.yaml"
    
    # Database
    RECEIPT_DB_URL = os.getenv("RECEIPT_DB_URL", "sqlite:///receipts.db")
    CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

    # Directories
    INBOX_DIR = OBSIDIAN_VAULT / "00_Skrzynka"
    TEMP_DIR = BASE_DIR / "temp_processing"
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # RAG Settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000))
    RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 200))

    # Prompts
    PROMPTS = {}
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            try:
                PROMPTS = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load prompts config: {e}")

    # Theme
    BRAIN_THEME = {
        "user": "bold green",
        "agent": "bold blue",
        "status_info": "dim cyan",
        "status_ok": "bold green",
        "status_warn": "bold yellow",
        "status_err": "bold red",
        "time": "dim white"
    }

    TAGS_CONFIG = {}
    if TAGS_FILE.exists():
        with open(TAGS_FILE, "r", encoding="utf-8") as f:
            try:
                TAGS_CONFIG = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load tags config: {e}")

    # Feature Flags
    SMART_ROUTER_ENABLED = os.getenv("SMART_ROUTER_ENABLED", "true").lower() == "true"

    # Categories Configuration
    CATEGORIES = [
        "Edukacja",
        "Newsy",
        "Badania",
        "Zasoby",
        "Dziennik",
        "Prywatne",
        "Przegląd",
        "Biznes",
        "Zgodność",
        "Cyberbezpieczeństwo",
        "Inżynieria_Danych",
        "Produktywność",
        "Technologie",
        "Paragony",
        "Zalaczniki"
    ]

    # Queue Settings
    BRAIN_GUARD_QUEUE_SIZE = int(os.getenv("BRAIN_GUARD_QUEUE_SIZE", 5))
    BRAIN_GUARD_MAX_WORKERS = int(os.getenv("BRAIN_GUARD_MAX_WORKERS", 2))
