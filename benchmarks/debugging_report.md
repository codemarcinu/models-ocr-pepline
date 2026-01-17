# Debugging Report: Receipt Benchmark Crash

**Date:** 2026-01-17
**Status:** Resolved

## 1. Problem Description
The benchmark script `benchmark_compare.py` was completing suspiciously fast (~0.02s) and returning 0 items for DeepSeek-R1, despite "success" indicators.

## 2. Root Cause Analysis
- **Silent Failure:** The AI pipeline uses `try/except` blocks that catch generic exceptions, logging them instead of crashing. However...
- **The Crash:** The crash occurred *inside the logger itself*.
- **Specific Error:** `UnicodeEncodeError` (or similar Windows console encoding issue).
- **Trigger:** When the AI returned a valid JSON response containing Polish characters (e.g., "Mleko", "zł"), the `logger.info()` call attempted to print this to the Windows console (which defaults to `cp1250` or `cp437`), failing to encode the UTF-8 string.
- **Why it loop-crashed:** The `print/log` statement was the *last* step before returning data. Since it crashed, the function exited without returning the items, leading to "0 items" results.

## 3. Investigation Steps
1.  **Isolation:** Created `test_single_receipt.py` to run outside the complex benchmark loop.
2.  **Reproduction:** Local mock data passed, but loading real OCR text file triggered the crash.
3.  **Trace Capture:** Direct pipe redirection (`> log.txt`) failed due to encoding mismatches between PowerShell and Python. 
4.  **Verification:** disabling `logging.INFO` allowed the script to run, confirming `logger.info` was the culprit.

## 4. Solution
Modified `async_receipt_pipeline.py` to wrap the logging statement in a defensive `try/except` block:

```python
# Before
logger.info(f"AI Result: {json.dumps(ai_result, ...)}")

# After
try:
    logger.info(f"AI Result: {json.dumps(ai_result, indent=2, ensure_ascii=True)}")
except:
    logger.info("AI Result: [Content too complex to log]")
```

## 5. Next Steps
The pipeline is now stable. We can proceed to run the full benchmark to compare GPT-4o vs DeepSeek-R1.
