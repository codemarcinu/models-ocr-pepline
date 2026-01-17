import sys
import os
import re
import json
import asyncio
from pathlib import Path

# Add current directory to sys.path
sys.path.append(os.getcwd())

from config import ProjectConfig
from utils.receipt_agents import detect_shop, get_agent
from adapters.google.gemini_adapter import UniversalBrain

def find_receipt_file(filename_part):
    """Finds the receipt file in the vault."""
    vault = ProjectConfig.OBSIDIAN_VAULT
    matches = list(vault.rglob(f"*{filename_part}*.md"))
    # Filter out potential .pdf or other files just in case regex matched them if logic was different
    matches = [m for m in matches if m.suffix == '.md']
    return matches[0] if matches else None

def extract_ocr_from_md(file_path):
    """Extracts OCR text from the markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decode failed, trying latin-1")
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()
    
    print(f"Content length: {len(content)}")
    if "Oryginalny OCR" not in content:
        print("❌ 'Oryginalny OCR' string NOT found in content!")
        print("Head of file:")
        print(content[:500])
        return None
        
    print("✅ 'Oryginalny OCR' string found. Attempting regex...")
    match = re.search(r'## 📜 Oryginalny OCR\s*\n(.*?)(?=\n#|\Z)', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    print("❌ Regex failed to match. Printing context around 'Oryginalny OCR':")
    idx = content.find("Oryginalny OCR")
    start = max(0, idx - 100)
    end = min(len(content), idx + 1000)
    print(content[start:end])
    return None

async def verify_extraction(file_path):
    print(f"--- Verifying: {file_path.name} ---")
    
    ocr_text = extract_ocr_from_md(file_path)
    if not ocr_text:
        print("❌ No OCR text found in file!")
        return

    print(f"Extracted OCR length: {len(ocr_text)} chars")
    
    shop = detect_shop(ocr_text)
    print(f"Detected Shop: {shop}")
    
    agent = get_agent(shop)
    processed_ocr = agent.preprocess(ocr_text)
    
    print("\n--- Preprocessed OCR (First 500 chars) ---")
    print(processed_ocr[:500])
    print("------------------------------------------\n")

    # Initialize Brain
    brain = UniversalBrain()
    print(f"Using Brain: {brain.provider}")
    
    # Run Extraction
    system_prompt = agent.get_prompt()
    user_prompt = f"""
    Sklep: {shop}
    Data: 2024-05-12 (Forced for verification)
    
    Surowy OCR:
    {processed_ocr[:12000]}
    
    Wyciągnij produkty jako JSON.
    """
    
    print("Sending to AI...")
    try:
        response = brain.generate_content(user_prompt, system_prompt=system_prompt, format="json")
        print("\n--- RAW AI RESPONSE ---")
        print(response)
        print("-----------------------\n")
        
        # Parse JSON
        if "```json" in response:
            json_text = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL).group(1)
        elif "```" in response:
             json_text = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL).group(1)
        else:
            json_text = response
            
        data = json.loads(json_text)
        items = data.get('items') or data.get('produkty') or []
        
        print(f"\n✅ AI Extracted {len(items)} items:")
        print(f"{'PRODUKT':<40} | {'ILOŚĆ'} | {'CENA'} | {'SUMA'}")
        print("-" * 70)
        
        for item in items:
            name = item.get('nazwa') or item.get('name')
            qty = item.get('ilosc') or item.get('qty') or item.get('quantity')
            price = item.get('cena_jedn') or item.get('price')
            total = item.get('suma') or item.get('sum') or item.get('total')
            
            print(f"{str(name)[:40]:<40} | {str(qty):<5} | {str(price):<6} | {str(total)}")
            
        # Check against OCR text availability (Basic Hallucination Check)
        print("\n--- Hallucination Check ---")
        hallucinations = []
        for item in items:
            name = item.get('nazwa') or item.get('name')
            # Normalize name for search
            name_parts = name.split()
            found_parts = 0
            for part in name_parts:
                if len(part) > 3 and part.lower() in ocr_text.lower():
                    found_parts += 1
            
            if found_parts < len(name_parts) / 2: # heuristic: if less than half words found
                hallucinations.append(name)
                print(f"⚠️ POTENTIAL HALLUCINATION: '{name}' not clearly found in OCR")
                
        if not hallucinations:
            print("✅ No obvious hallucinations detected (based on text overlap).")

    except Exception as e:
        print(f"❌ Error during AI processing: {e}")

if __name__ == "__main__":
    print(f"Vault Path: {ProjectConfig.OBSIDIAN_VAULT}", flush=True)
    target_file = find_receipt_file("2024-05-12_Paragon_Biedronka_Zakupy")
    print(f"Target found: {target_file}", flush=True)
    
    if target_file:
        asyncio.run(verify_extraction(target_file))
    else:
        print("File not found.", flush=True)
