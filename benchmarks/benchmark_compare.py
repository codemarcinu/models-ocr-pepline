import asyncio
import os
import time
import json
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
import sys
import logging

# Set logging to ERROR to keep output clean
logging.basicConfig(level=logging.INFO)

sys.path.append('.')

from async_receipt_pipeline import AsyncReceiptPipeline
from adapters.google.gemini_adapter import UniversalBrain
from core.ocr_service import GoogleVisionOCR
from config import ProjectConfig

async def run_comparison():
    # Setup paths
    images_dir = Path("benchmarks/images")
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return

    # Gather images
    extensions = {'.jpg', '.jpeg', '.png'}
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in extensions]
    # images = images[:1] # DEBUG: Run only 1 image
    
    if not images:
        print("❌ No images found in benchmarks/images/")
        return

    print(f"🚀 Starting Comparison Benchmark on {len(images)} receipts...")
    print(f"MODELS: gpt-4o-mini vs deepseek-r1:latest")

    # Initialize Services
    try:
        ocr_service = GoogleVisionOCR()
        # Trigger client init to check creds early
        _ = ocr_service.client 
    except Exception as e:
        print(f"❌ OCR Service Error: {e}")
        return

    pipeline_brain_swap = AsyncReceiptPipeline()
    
    # Brains
    brain_openai = UniversalBrain(provider="openai")
    brain_local = UniversalBrain(provider="local")

    results = []
    
    total_start = time.time()

    for img_path in images:
        print(f"\n📸 Processing: {img_path.name}")
        
        # 1. OCR Step
        ocr_start = time.time()
        ocr_text = ocr_service.detect_text(img_path)
        ocr_time = time.time() - ocr_start
        
        if not ocr_text:
            print("  ❌ OCR Failed or empty")
            continue
            
        print(f"  ✅ OCR: {len(ocr_text)} chars ({ocr_time:.2f}s)")

        # 2. GPT-4o-mini Run
        print("  🤖 Running gpt-4o-mini...", end="", flush=True)
        pipeline_brain_swap.brain = brain_openai
        start_gpt = time.time()
        try:
            res_gpt = await pipeline_brain_swap.process_receipt_async(
                ocr_text, 
                model_name="gpt-4o-mini",
                force_ai=True
            )
            time_gpt = time.time() - start_gpt
            items_gpt = len(res_gpt.get('items', []))
            total_gpt = res_gpt.get('total_amount', 0.0)
            status_gpt = "✅" if items_gpt > 0 else "⚠️"
            print(f" Done ({time_gpt:.2f}s, {items_gpt} items)")
        except Exception as e:
            print(f" Failed: {e}")
            time_gpt = 0
            items_gpt = 0
            total_gpt = 0
            status_gpt = "❌"

        # 3. DeepSeek-R1 Run
        print("  🧠 Running deepseek-r1...", end="", flush=True)
        pipeline_brain_swap.brain = brain_local
        start_ds = time.time()
        try:
            res_ds = await pipeline_brain_swap.process_receipt_async(
                ocr_text, 
                model_name="deepseek-r1:latest",
                force_ai=True
            )
            time_ds = time.time() - start_ds
            items_ds = len(res_ds.get('items', []))
            total_ds = res_ds.get('total_amount', 0.0)
            status_ds = "✅" if items_ds > 0 else "⚠️"
            print(f" Done ({time_ds:.2f}s, {items_ds} items)")
            
            # Save DeepSeek detailed result for analysis
            debug_file = img_path.with_suffix(".deepseek.json")
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(res_ds, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f" Failed: {e}")
            time_ds = 0
            items_ds = 0
            total_ds = 0
            status_ds = "❌"

        results.append({
            "File": img_path.name,
            "GPT Items": items_gpt,
            "DS Items": items_ds,
            "GPT Total": total_gpt,
            "DS Total": total_ds,
            "GPT Time": round(time_gpt, 2),
            "DS Time": round(time_ds, 2)
        })

    pipeline_brain_swap.close()
    
    # Generate Report
    total_elapsed = time.time() - total_start
    
    report_lines = []
    report_lines.append(f"# Benchmark Comparison: GPT-4o vs DeepSeek-R1")
    report_lines.append(f"Date: {datetime.now().isoformat()}")
    report_lines.append("")
    report_lines.append(tabulate(results, headers="keys", tablefmt="github"))
    report_lines.append("")
    report_lines.append(f"Total Benchmark Time: {total_elapsed:.2f}s")
    
    report_content = "\n".join(report_lines)
    print("\n" + report_content)

    with open("benchmark_comparison.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print("\n✅ Saved to benchmark_comparison.md")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        asyncio.run(run_comparison())
    except KeyboardInterrupt:
        print("\nCancelled.")
