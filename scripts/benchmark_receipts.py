
import asyncio
import json
import logging
import time
import re
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
from config import ProjectConfig

# Difficult receipt with noise, abbreviations, and missing context
DIFFICULT_RECEIPT = """
B!EDRONKA NR 2137
u1. Jp2 21, Wadowice
777-777-77-77

PARAGON FISKALNY
2026-01-14 18:30

BUŁKA GRAH. 70G         7x0.99  6.93 A
SER. ŻÓŁ. GOU.          18.99 A
MLEK. ŚW. 2% 1L          4.49 D
WODA MIN. NG 1,5L       6x1.99 11.94 A
POMID. MALIN. LUZ        1.255*12.99 16.30 C
KIEŁB. ŚLĄSKA            23.90 B
RABAT - KIEŁB. ŚLĄSKA   -4.00

SUMA PLN                       78.55
Rozliczenie płatności:
Inna                          78.55
001456/1234  #1

NIP NABYW: 555-555-55-55
"""

# Parsing logic for DeepSeek's <think> tags
def clean_deepseek_output(text: str) -> str:
    """Removes distinct <think>...<think> block."""
    # Remove <think> content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove markdown code blocks if present
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

async def benchmark_model(model_name: str, ocr_text: str):
    """Run extraction benchmark on a model."""
    print(f"\n--- Testing Model: {model_name} ---")
    
    system_prompt = """Jesteś ekspertem od paragonów. Wyciągnij dane w JSON:
    {
      "items": [
        {"nazwa": "str", "ilosc": float, "cena_jedn": float, "suma": float}
      ]
    }
    """
    
    user_prompt = f"Paragon:\n{ocr_text}\n\nWyciągnij JSON."
    
    start_time = time.time()
    try:
        if "deepseek" in model_name:
            # DeepSeek often needs a slight push to not just 'think' but output proper JSON provided
            # Standard prompt usually works but let's see
            pass

        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            stream=False
        )
        
        elapsed = time.time() - start_time
        raw_content = response['message']['content']
        
        # Post-processing
        cleaned_content = raw_content
        if "<think>" in raw_content:
            print(f"[{model_name}] Detected <think> tags. Raw length: {len(raw_content)}")
            cleaned_content = clean_deepseek_output(raw_content)
        
        print(f"[{model_name}] Time: {elapsed:.2f}s")
        
        # Validation
        try:
            data = json.loads(cleaned_content)
            items = data.get('items', [])
            print(f"[{model_name}] Extracted Items: {len(items)}")
            # Print items for visual verification
            for item in items:
                print(f"  - {item}")
            
            # Score (simple heuristic)
            score = 0
            expected_items = 6 # Bułka, Ser, Mleko, Woda, Pomidor, Kiełbasa
            if abs(len(items) - expected_items) <= 1:
                score += 50
            
            # Check for tricky item parsing (e.g. Pomidor quantity)
            for item in items:
                if "POMID" in item.get('nazwa', '').upper() and item.get('ilosc') == 1.255:
                    score += 50
                    print(f"[{model_name}] ✅ Correctly parsed complex quantity (1.255)")
            
            print(f"[{model_name}] Score: {score}/100")
            return score, elapsed
            
        except json.JSONDecodeError:
            print(f"[{model_name}] ❌ Invalid JSON output")
            print(f"Preview: {cleaned_content[:200]}...")
            return 0, elapsed

    except Exception as e:
        print(f"[{model_name}] Error: {e}")
        return 0, 0

async def main():
    # Models to test
    # 1. Current Polish Model
    bielik_model = ProjectConfig.OLLAMA_GENERATION_MODEL
    # 2. DeepSeek R1 (assuming user has pulled it or we pull it)
    deepseek_model = "deepseek-r1:latest"
    
    # Ensure deepseek is pulled
    # We assume usage of run_command previously to verify, but here we just try
    
    results = {}
    
    # Run Bielik
    try:
        score_b, time_b = await benchmark_model(bielik_model, DIFFICULT_RECEIPT)
        results[bielik_model] = {"score": score_b, "time": time_b}
    except Exception as e:
         print(f"Bielik failed: {e}")

    # Run DeepSeek
    try:
        score_d, time_d = await benchmark_model(deepseek_model, DIFFICULT_RECEIPT)
        results[deepseek_model] = {"score": score_d, "time": time_d}
    except Exception as e:
        print(f"DeepSeek failed: {e}")
        
    print("\n\n=== VERDICT ===")
    winner = max(results, key=lambda k: results[k]['score'])
    print(f"Winner: {winner}")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
