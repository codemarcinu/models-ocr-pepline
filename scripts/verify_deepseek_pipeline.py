
import asyncio
import logging
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from async_receipt_pipeline import AsyncReceiptPipeline
from config import ProjectConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

# OCR Text causing reasoning
DIFFICULT_OCR = """
B!EDRONKA NR 2137
Data: 2026-01-14

BUŁKA GRAH. 70G         7x0.99  6.93 A
DZIWNY PRODUKT          10.00 A

SUMA PLN                       16.93
"""

async def verify():
    # Force AI usage by ensuring cache miss
    pipeline = AsyncReceiptPipeline()
    
    # Mock cache to always return nothing for this test
    # (Or just use a unique product name)
    
    logger.info(f"Target Model: {ProjectConfig.OLLAMA_RECEIPT_MODEL}")
    
    try:
        result = await pipeline.process_receipt_async(DIFFICULT_OCR, shop="BIEDRONKA")
        
        logger.info("Pipeline finished.")
        logger.info(f"Items found: {len(result['items'])}")
        for item in result['items']:
            logger.info(f" - {item['nazwa']}: {item['suma']} PLN")
            
        if len(result['items']) == 2:
            logger.info("✅ Verification SUCCESS: 2 items extracted.")
        else:
            logger.warning(f"⚠️ Verification PARTIAL: Expected 2 items, got {len(result['items'])}.")
            
    except Exception as e:
        logger.error(f"❌ Verification FAILED: {e}")
    finally:
        pipeline.close()

if __name__ == "__main__":
    asyncio.run(verify())
