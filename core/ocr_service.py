import os
import io
import hashlib
from pathlib import Path
from typing import Optional
from google.cloud import vision
from config import ProjectConfig, logger

class GoogleVisionOCR:
    """
    Handles OCR using Google Cloud Vision API.
    Caches results to filesystem to minimize API costs/latency.
    """

    def __init__(self):
        self._client = None
        self.credentials_path = ProjectConfig.GOOGLE_APPLICATION_CREDENTIALS
        
    @property
    def client(self):
        if self._client is None:
            if not self.credentials_path.exists():
                raise FileNotFoundError(f"GCP Credentials not found at: {self.credentials_path}")
            
            self._client = vision.ImageAnnotatorClient()
        return self._client

    def detect_text(self, image_path: Path) -> Optional[str]:
        """
        Detects text in an image file. 
        Returns full text or None if failed.
        Checks for cached .txt file first.
        """
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        # Check cache
        cache_path = image_path.with_suffix(".txt")
        if cache_path.exists():
            logger.info(f"Using cached OCR for: {image_path.name}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()

        logger.info(f"Calling Google Vision API for: {image_path.name}")
        try:
            with io.open(image_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f'{response.error.message}')

            texts = response.text_annotations
            if texts:
                full_text = texts[0].description
                
                # Save to cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(full_text)
                    
                return full_text
            
            return ""

        except Exception as e:
            logger.error(f"OCR Failed for {image_path.name}: {e}")
            return None
