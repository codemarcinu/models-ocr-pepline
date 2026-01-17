from config import ProjectConfig
import logging

print(f"StreamHandler Level: {logging.getLogger().handlers[0].level} (Expected: {logging.ERROR})")
print(f"urllib3 Level: {logging.getLogger('urllib3').level} (Expected: {logging.ERROR})")
print(f"chromadb Level: {logging.getLogger('chromadb').level} (Expected: {logging.ERROR})")
print(f"sentence_transformers Level: {logging.getLogger('sentence_transformers').level} (Expected: {logging.WARNING})")
print(f"backoff Level: {logging.getLogger('backoff').level} (Expected: {logging.ERROR})")
