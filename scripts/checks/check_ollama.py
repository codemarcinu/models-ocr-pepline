import ollama
import sys
from config import ProjectConfig, logger

def check_ollama():
    model_name = ProjectConfig.OLLAMA_MODEL
    print(f"🔍 Sprawdzanie modelu Ollama: {model_name}...")
    
    try:
        # Check if Ollama is running by listing models
        response = ollama.list()
        
        # Newer versions of ollama-python use objects, older use dicts
        available_models = []
        models_list = response.get('models', []) if isinstance(response, dict) else getattr(response, 'models', [])
        
        for m in models_list:
            if isinstance(m, dict):
                available_models.append(m.get('name', m.get('model', '')))
            else:
                available_models.append(getattr(m, 'name', getattr(m, 'model', '')))
        
        # Normalize names (e.g., 'bielik:latest' matches 'bielik')
        is_available = any(model_name in m for m in available_models if m)
        
        if is_available:
            print(f"✅ Model '{model_name}' jest dostępny.")
            return True
        else:
            print(f"❌ Model '{model_name}' NIE ZNALEZIONY.")
            print(f"⚠️  Aby naprawić, uruchom w nowym terminalu:  ollama pull {model_name}")
            return False

    except Exception as e:
        print(f"❌ Nie można połączyć się z Ollama.")
        print(f"   Błąd: {e}")
        print("⚠️  Upewnij się, że Ollama jest zainstalowana i uruchomiona (komenda 'ollama serve').")
        return False

if __name__ == "__main__":
    success = check_ollama()
    if not success:
        # We exit with 0 to not block startup, but we warned the user.
        # Or we can exit with 1 to force them to fix it.
        # Let's exit with 0 but give a strong warning.
        print("\n⚠️  Aplikacja uruchomi się, ale funkcje AI (generowanie notatek) mogą nie działać.")
        sys.exit(0)
