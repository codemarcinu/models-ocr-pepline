import sys
import os
from pathlib import Path

# Add current directory to sys.path
sys.path.append(os.getcwd())

from config import ProjectConfig

def run():
    matches = list(ProjectConfig.OBSIDIAN_VAULT.rglob('*2024-05-12_Paragon_Biedronka_Zakupy.md*'))
    if not matches:
        print("File not found")
        return
        
    path = matches[0]
    print(f"Reading: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
         with open(path, 'r', encoding='latin-1') as f:
            content = f.read()
            
    print("--- CONTENT START ---")
    print(content[:1500])
    print("--- CONTENT END ---")

if __name__ == "__main__":
    run()
