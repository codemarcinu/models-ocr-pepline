
import ollama
import logging
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import ProjectConfig

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Verification")

def verify_models():
    """Verify that the configured models are working correctly."""
    
    embedding_model = ProjectConfig.OLLAMA_EMBEDDING_MODEL
    generation_model = ProjectConfig.OLLAMA_GENERATION_MODEL
    
    logger.info(f"Target Embedding Model: {embedding_model}")
    logger.info(f"Target Generation Model: {generation_model}")
    
    # 1. Check if models are available
    try:
        models = ollama.list()
        logger.info(f"Ollama list response keys: {models.keys()}")
        if 'models' in models:
             # Debug first model structure
             if len(models['models']) > 0:
                 logger.info(f"First model sample: {models['models'][0]}")
             
             # Handle different potential structures
             model_names = []
             for m in models['models']:
                 if 'name' in m:
                     model_names.append(m['name'])
                 elif 'model' in m:
                     model_names.append(m['model'])
                 else:
                     logger.warning(f"Unknown model structure: {m}")

             # Check for embedding model (partial match)
             emb_found = any(embedding_model in m for m in model_names)
             if not emb_found:
                  logger.warning(f"⚠️ Embedding model {embedding_model} not found in list. Available: {model_names}")
             else:
                  logger.info(f"✅ Embedding model {embedding_model} found.")
                  
             # Check for generation model
             gen_found = any(generation_model in m for m in model_names)
             if not gen_found:
                  logger.warning(f"⚠️ Generation model {generation_model} not found in list.")
             else:
                  logger.info(f"✅ Generation model {generation_model} found.")
        else:
            logger.error(f"Unexpected ollama.list() structure: {models}")
             
    except Exception as e:
        logger.error(f"Failed to list models (skipping check): {e}")

    # 2. Test Embeddings
    logger.info("Testing Embedding Generation...")
    try:
        text = "To jest testowy tekst po polsku."
        response = ollama.embeddings(model=embedding_model, prompt=text)
        if 'embedding' in response and len(response['embedding']) > 0:
            logger.info(f"✅ Embeddings generated successfully. Vector length: {len(response['embedding'])}")
        else:
            logger.error("❌ Failed to generate embeddings.")
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")

    # 3. Test Polish Generation
    logger.info("Testing Polish Generation...")
    try:
        prompt = "Opowiedz krótko o Systemie Słonecznym. Odpowiedz tylko po polsku."
        response = ollama.chat(
            model=generation_model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        content = response['message']['content']
        logger.info(f"Response:\n{content}\n")
        
        # Simple heuristic check for Polish
        if any(char in content for char in "ąęśćżźółń"):
            logger.info("✅ Response likely in Polish (Polish chars detected).")
        else:
            logger.warning("⚠️ Response might not be in Polish.")
            
    except Exception as e:
        logger.error(f"❌ Error generating text: {e}")

if __name__ == "__main__":
    verify_models()
