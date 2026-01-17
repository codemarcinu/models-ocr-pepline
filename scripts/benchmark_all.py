
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
from tqdm import tqdm

# --- CONFIGURATION ---
MODELS = [
    "SpeakLeash/bielik-11b-v2.3-instruct:Q5_K_M",
    "deepseek-r1:latest",
    "qwen2.5:14b",
    "llama3.2:latest",
    "gemma3:4b",
    # "gpt-oss:latest" # Uncomment if verifiable
]

# --- DATASETS ---

# 1. RECEIPT (Messy OCR - High Difficulty)
DATA_RECEIPT = """
B!EDRONKA NR 2137
u1. Jp2 21, Wadowice
Data: 2026-01-14 18:30

BUŁKA GRAH. 70G         7x0.99  6.93 A
SER. ŻÓŁ. GOU.          18.99 A
MLEK. ŚW. 2% 1L          4.49 D
WODA MIN. NG 1,5L       6x1.99 11.94 A
POMID. MALIN. LUZ        1.255*12.99 16.30 C
KIEŁB. ŚLĄSKA            23.90 B
RABAT - KIEŁB. ŚLĄSKA   -4.00

SUMA PLN                       78.55
"""

PROMPT_RECEIPT_SYS = """Jesteś ekspertem OCR. Wyciągnij JSON: {"items": [{"nazwa": "str", "ilosc": float, "cena_jedn": float, "suma": float}]}"""
PROMPT_RECEIPT_USER = f"Paragon:\n{DATA_RECEIPT}\n\nJSON:"

# 2. ARTICLE (Technical - Summarization)
DATA_ARTICLE = """
**Tytuł: Wpływ Architektury Mixture of Experts (MoE) na Efektywność LLM**

W ostatnich latach modele językowe (LLM) stały się fundamentem nowoczesnej sztucznej inteligencji. Jednakże, ich rosnąca skala wiąże się z ogromnymi kosztami obliczeniowymi. Odpowiedzią na ten problem jest architektura Mixture of Experts (MoE), spopularyzowana m.in. przez model Mixtral 8x7B.

W klasycznym modelu (dense), każde zapytanie aktywuje 100% parametrów sieci. Jest to nieefektywne, gdyż proste pytania nie wymagają pełnej mocy modelu. Architektura MoE wprowadza koncepcję "ekspertów" – wyspecjalizowanych podsieci. Router (bramka) decyduje, którzy eksperci (zazwyczaj 1 lub 2) powinni przetworzyć dany token.

Dzięki temu, model może mieć np. 47 miliardów parametrów, ale podczas inferencji używać tylko 13 miliardów (tzw. active parameters). Pozwala to na drastyczne zwiększenie przepustowości (tokens/sec) i obniżenie kosztów VRAM, przy zachowaniu jakości modelu "dense" o znacznie większym rozmiarze.

Kluczowym wyzwaniem MoE jest trening routera – musi on unikać sytuacji, w której jeden ekspert jest przeciążony, a inni bezczynni (load balancing). Nowoczesne techniki, takie jak expert parallelism, pozwalają trenować te modele na klastrach GPU z wysoką efektywnością.
"""

PROMPT_ARTICLE_SYS = """Jesteś ekspertem AI. Streść podany tekst w 3 punktach po polsku."""
PROMPT_ARTICLE_USER = f"Tekst:\n{DATA_ARTICLE}\n\nStreszczenie:"

# 3. TRANSCRIPT (Meeting - Action Items)
DATA_TRANSCRIPT = """
[00:00:15] Marek: Dobra, zaczynamy. Celem spotkania jest ustalenie roadmapy na Q1. Ania, jak tam backend?
[00:00:22] Ania: Migracja bazy danych zakończona. Ale mamy problem z API do płatności, wywala 500 przy dużym obciążeniu.
[00:00:30] Marek: Okej, to priorytet. Tomek, dasz radę na to zerknąć do piątku?
[00:00:35] Tomek: Mam teraz sprint z mobilem, ale mogę w czwartek usiąść.
[00:00:40] Marek: Dobra. A co z marketingiem?
[00:00:43] Kasia: Kampania rusza w poniedziałek. Potrzebuję tylko ostatecznych grafik od designu.
[00:00:48] Marek: Zanotowane. Czyli podsumowując: Ania testuje bazę, Tomek naprawia API w czwartek, Kasia ciśnie grafików na poniedziałek.
"""

PROMPT_TRANSCRIPT_SYS = """Wymień zadania (Action Items) z transkrypcji w formacie JSON: {"tasks": [{"kto": "str", "zadanie": "str", "termin": "str"}]}."""
PROMPT_TRANSCRIPT_USER = f"Transkrypcja:\n{DATA_TRANSCRIPT}\n\nJSON:"


# --- UTILS ---

def clean_deepseek_output(text: str) -> str:
    """Removes distinct <think>...<think> block and markdown formatting."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

async def score_json_match(text: str, expected_key_count: int, key_name: str) -> int:
    score = 0
    try:
        data = json.loads(text)
        items = data.get(key_name, [])
        
        # Quantity Score
        if abs(len(items) - expected_key_count) <= 1:
            score += 50
        elif abs(len(items) - expected_key_count) <= 2:
            score += 25
            
        # Validity Score
        if len(items) > 0:
            score += 50
            
        return score
    except:
        return 0

async def score_text_summary(text: str) -> int:
    score = 0
    # Check language (simple Polish words)
    if any(w in text.lower() for w in ["jest", "są", "model", "eksperci", "dzięki", "moe"]):
        score += 50
    # Check format (bullets)
    if text.count("- ") >= 2 or text.count("1.") >= 2:
        score += 50
    return score

async def run_test(model, task_name, sys_prompt, user_prompt, scorer_func):
    start = time.time()
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            stream=False,
            options={'temperature': 0.1} # Low temp for deterministic logic
        )
        elapsed = time.time() - start
        content = response['message']['content']
        
        # DeepSeek Cleaning
        if "deepseek" in model.lower():
            content = clean_deepseek_output(content)
            
        score = await scorer_func(content)
        
        return {
            "time": elapsed,
            "score": score,
            "sample": content[:100].replace('\n', ' ') + "..."
        }
    except Exception as e:
        return {"time": 0, "score": 0, "sample": f"ERROR: {str(e)[:50]}"}

async def main():
    print(f"{'MODEL':<45} | {'TASK':<15} | {'SCORE':<5} | {'TIME (s)':<8} | {'SAMPLE'}")
    print("-" * 120)
    
    results_map = {}

    for model in MODELS:
        results_map[model] = {}
        
        # 1. Receipt Test
        res_r = await run_test(
            model, "Receipt", PROMPT_RECEIPT_SYS, PROMPT_RECEIPT_USER,
            lambda t: score_json_match(t, 6, "items")
        )
        print(f"{model:<45} | {'Receipt':<15} | {res_r['score']:<5} | {res_r['time']:<8.2f} | {res_r['sample']}")
        
        # 2. Article Test
        res_a = await run_test(
            model, "Article", PROMPT_ARTICLE_SYS, PROMPT_ARTICLE_USER,
            score_text_summary
        )
        print(f"{model:<45} | {'Article':<15} | {res_a['score']:<5} | {res_a['time']:<8.2f} | {res_a['sample']}")

        # 3. Transcript Test
        res_t = await run_test(
            model, "Transcript", PROMPT_TRANSCRIPT_SYS, PROMPT_TRANSCRIPT_USER,
            lambda t: score_json_match(t, 3, "tasks")
        )
        print(f"{model:<45} | {'Transcript':<15} | {res_t['score']:<5} | {res_t['time']:<8.2f} | {res_t['sample']}")
        print("-" * 120)

if __name__ == "__main__":
    asyncio.run(main())
