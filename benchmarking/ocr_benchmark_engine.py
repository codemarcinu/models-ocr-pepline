#!/usr/bin/env python3
"""
OCR Benchmarking Engine
Compares Google Vision API, GPT-4o mini, and DeepSeek R1 for receipt OCR extraction.
"""

import json
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRProvider(str, Enum):
    """Supported OCR providers."""
    GOOGLE_VISION = "google_vision"
    GPT4O_MINI = "gpt4o_mini"
    DEEPSEEK_R1 = "deepseek_r1"


@dataclass
class OCRResult:
    """OCR extraction result."""
    provider: OCRProvider
    receipt_id: str
    raw_text: str
    extracted_data: Dict[str, Any]
    processing_time: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = None
    model_version: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkMetrics:
    """Benchmark comparison metrics."""
    receipt_id: str
    provider: OCRProvider
    
    # Accuracy metrics
    field_accuracy: float  # 0-1 exact match rate
    fuzzy_accuracy: float  # 0-1 fuzzy match rate (>80% similarity)
    char_error_rate: float  # 0-1 character-level error rate
    word_error_rate: float  # 0-1 word-level error rate
    
    # Performance metrics
    processing_time: float  # seconds
    tokens_used: Optional[int] = None
    cost_per_page: Optional[float] = None
    
    # Data quality
    field_completeness: float  # percentage of expected fields extracted
    numerical_accuracy: float  # accuracy for numerical fields (amounts, quantities)
    
    # Business logic
    consistency_score: float  # validation against business rules
    
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class OCRExtractor(ABC):
    """Abstract base class for OCR providers."""

    def __init__(self, provider: OCRProvider):
        self.provider = provider

    @abstractmethod
    def extract(self, image_path: str) -> Tuple[str, Dict[str, Any], float]:
        """
        Extract text and data from image.
        
        Returns:
            Tuple of (raw_text, extracted_data_dict, processing_time)
        """
        pass

    @abstractmethod
    def get_cost(self, tokens_used: int) -> float:
        """Calculate cost for API call."""
        pass


class GoogleVisionExtractor(OCRExtractor):
    """Google Cloud Vision API extractor."""

    def __init__(self):
        super().__init__(OCRProvider.GOOGLE_VISION)
        try:
            from google.cloud import vision
            self.client = vision.ImageAnnotatorClient()
        except ImportError:
            logger.error("google-cloud-vision not installed")
            self.client = None

    def extract(self, image_path: str) -> Tuple[str, Dict[str, Any], float]:
        """Extract using Google Vision API."""
        if self.client is None:
            raise RuntimeError("Google Vision client not initialized")

        start_time = time.time()

        with open(image_path, "rb") as image_file:
            content = image_file.read()

        image = {"content": content}
        request = {"image": image, "features": [{"type_": 1, "max_results": 10}]}

        response = self.client.document_text_detection(request)
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""

        processing_time = time.time() - start_time

        # Parse structured data from Vision response
        extracted_data = self._parse_receipt(full_text, response)

        return full_text, extracted_data, processing_time

    def _parse_receipt(self, text: str, response: Any) -> Dict[str, Any]:
        """Parse receipt fields from Vision response."""
        # This is a placeholder - implement actual parsing logic
        return {
            "merchant_name": "",
            "date": "",
            "time": "",
            "total_amount": 0.0,
            "items": [],
            "raw_text": text
        }

    def get_cost(self, tokens_used: int) -> float:
        """Google Vision API pricing (2024): ~$0.0015 per feature/image."""
        return 0.0015


class GPT4oMiniExtractor(OCRExtractor):
    """OpenAI GPT-4o mini extractor."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(OCRProvider.GPT4O_MINI)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            logger.error("openai not installed")
            self.client = None

    def extract(self, image_path: str) -> Tuple[str, Dict[str, Any], float]:
        """Extract using GPT-4o mini vision."""
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        start_time = time.time()

        # Encode image
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Call GPT-4o mini with vision
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                        {
                            "type": "text",
                            "text": self._get_extraction_prompt()
                        }
                    ],
                }
            ],
            max_tokens=2048,
        )

        processing_time = time.time() - start_time
        raw_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        # Parse JSON response
        extracted_data = self._parse_response(raw_text)
        extracted_data["tokens_used"] = tokens_used

        return raw_text, extracted_data, processing_time

    def _get_extraction_prompt(self) -> str:
        """Get extraction prompt for GPT-4o mini."""
        return """Extract receipt data and return as JSON with these fields:
{
  "merchant_name": "store name",
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "total_amount": 0.00,
  "tax_amount": 0.00,
  "items": [
    {"description": "item name", "quantity": 1, "unit_price": 0.00, "total": 0.00}
  ],
  "payment_method": "cash/card"
}
Return ONLY valid JSON."""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from GPT-4o mini."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return {
            "merchant_name": "",
            "date": "",
            "time": "",
            "total_amount": 0.0,
            "items": [],
            "raw_response": response_text
        }

    def get_cost(self, tokens_used: int) -> float:
        """GPT-4o mini pricing (2024): $0.00015 per input token, $0.0006 per output token."""
        # Assume ~60% input, 40% output split
        input_tokens = int(tokens_used * 0.6)
        output_tokens = tokens_used - input_tokens
        return (input_tokens * 0.00015) + (output_tokens * 0.0006)


class DeepSeekR1Extractor(OCRExtractor):
    """DeepSeek R1 local LLM extractor."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(OCRProvider.DEEPSEEK_R1)
        self.model_path = model_path or "deepseek-ai/DeepSeek-OCR"
        self.client = None
        self._init_model()

    def _init_model(self):
        """Initialize DeepSeek R1 model."""
        try:
            import ollama
            self.client = ollama.Client()
            logger.info(f"DeepSeek R1 initialized via Ollama")
        except ImportError:
            logger.warning("ollama not installed, trying vLLM...")
            try:
                from vllm import LLM
                self.client = LLM(model=self.model_path, tensor_parallel_size=1)
                logger.info(f"DeepSeek R1 initialized via vLLM")
            except ImportError:
                logger.error("Neither ollama nor vLLM installed")

    def extract(self, image_path: str) -> Tuple[str, Dict[str, Any], float]:
        """Extract using DeepSeek R1."""
        if self.client is None:
            raise RuntimeError("DeepSeek R1 not initialized")

        start_time = time.time()

        # Encode image
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = self._get_extraction_prompt()

        # Call model
        try:
            import ollama
            if isinstance(self.client, ollama.Client):
                response = self.client.generate(
                    model="deepseek-r1",
                    prompt=prompt,
                    images=[image_data],
                )
                raw_text = response["response"]
        except:
            raw_text = self._fallback_extract(image_path, prompt)

        processing_time = time.time() - start_time
        extracted_data = self._parse_response(raw_text)

        return raw_text, extracted_data, processing_time

    def _get_extraction_prompt(self) -> str:
        """Get extraction prompt optimized for DeepSeek R1."""
        return """Analyze this receipt image and extract structured data.
Provide output in JSON format:
{
  "merchant_name": "name",
  "date": "YYYY-MM-DD",
  "total_amount": 0.00,
  "items": [{"description": "...", "quantity": 1, "price": 0.00}]
}
Focus on accuracy and complete all fields."""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse response from DeepSeek R1."""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        
        return {
            "merchant_name": "",
            "total_amount": 0.0,
            "items": [],
            "raw_response": response_text
        }

    def _fallback_extract(self, image_path: str, prompt: str) -> str:
        """Fallback OCR using Tesseract if model unavailable."""
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(image_path)
            return pytesseract.image_to_string(img)
        except:
            logger.warning("Tesseract fallback failed")
            return ""

    def get_cost(self, tokens_used: int) -> float:
        """DeepSeek R1 local model - no direct cost, compute cost approximation."""
        # Approximate $0.00001 per token for local inference (energy cost)
        return tokens_used * 0.00001


class OCRBenchmark:
    """Main benchmarking orchestrator."""

    def __init__(self, ground_truth_dir: Path):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.results: List[OCRResult] = []
        self.metrics: List[BenchmarkMetrics] = []
        self.extractors = {}

    def register_extractor(self, provider: OCRProvider, extractor: OCRExtractor):
        """Register an OCR provider."""
        self.extractors[provider] = extractor
        logger.info(f"Registered {provider.value}")

    def run_benchmark(
        self,
        image_dir: Path,
        providers: Optional[List[OCRProvider]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run benchmarking suite."""
        if providers is None:
            providers = list(self.extractors.keys())

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        image_dir = Path(image_dir)
        images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

        logger.info(f"Running benchmark on {len(images)} images with {len(providers)} providers")

        for image_path in images:
            receipt_id = image_path.stem
            logger.info(f"Processing {receipt_id}")

            for provider in providers:
                if provider not in self.extractors:
                    logger.warning(f"Provider {provider.value} not registered")
                    continue

                try:
                    extractor = self.extractors[provider]
                    raw_text, extracted_data, processing_time = extractor.extract(str(image_path))

                    result = OCRResult(
                        provider=provider,
                        receipt_id=receipt_id,
                        raw_text=raw_text,
                        extracted_data=extracted_data,
                        processing_time=processing_time,
                        tokens_used=extracted_data.get("tokens_used"),
                        cost=extractor.get_cost(extracted_data.get("tokens_used", 0)),
                    )
                    self.results.append(result)

                    # Calculate metrics
                    metrics = self._calculate_metrics(result, receipt_id)
                    self.metrics.append(metrics)

                except Exception as e:
                    logger.error(f"Error processing {receipt_id} with {provider.value}: {e}")
                    result = OCRResult(
                        provider=provider,
                        receipt_id=receipt_id,
                        raw_text="",
                        extracted_data={},
                        processing_time=0,
                        error=str(e),
                    )
                    self.results.append(result)

        # Generate summary
        summary = self._generate_summary()

        if output_dir:
            self._save_results(output_dir, summary)

        return summary

    def _calculate_metrics(
        self,
        result: OCRResult,
        receipt_id: str
    ) -> BenchmarkMetrics:
        """Calculate benchmark metrics for a result."""
        ground_truth_file = self.ground_truth_dir / f"{receipt_id}.json"
        
        if ground_truth_file.exists():
            with open(ground_truth_file) as f:
                ground_truth = json.load(f)
        else:
            ground_truth = {}

        # Calculate accuracy metrics
        field_accuracy = self._calculate_field_accuracy(
            result.extracted_data,
            ground_truth
        )
        
        fuzzy_accuracy = self._calculate_fuzzy_accuracy(
            result.extracted_data,
            ground_truth
        )
        
        char_error_rate = self._calculate_char_error_rate(
            result.raw_text,
            ground_truth.get("raw_text", "")
        )
        
        word_error_rate = self._calculate_word_error_rate(
            result.raw_text,
            ground_truth.get("raw_text", "")
        )
        
        field_completeness = self._calculate_field_completeness(
            result.extracted_data
        )
        
        numerical_accuracy = self._calculate_numerical_accuracy(
            result.extracted_data,
            ground_truth
        )
        
        consistency_score = self._calculate_consistency(
            result.extracted_data
        )

        return BenchmarkMetrics(
            receipt_id=receipt_id,
            provider=result.provider,
            field_accuracy=field_accuracy,
            fuzzy_accuracy=fuzzy_accuracy,
            char_error_rate=char_error_rate,
            word_error_rate=word_error_rate,
            processing_time=result.processing_time,
            tokens_used=result.tokens_used,
            cost_per_page=result.cost,
            field_completeness=field_completeness,
            numerical_accuracy=numerical_accuracy,
            consistency_score=consistency_score,
        )

    def _calculate_field_accuracy(self, extracted: Dict, ground_truth: Dict) -> float:
        """Calculate exact match accuracy for fields."""
        if not ground_truth:
            return 0.0

        matches = 0
        total = 0

        for key in ["merchant_name", "date", "total_amount"]:
            if key in ground_truth:
                total += 1
                if extracted.get(key) == ground_truth[key]:
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _calculate_fuzzy_accuracy(self, extracted: Dict, ground_truth: Dict) -> float:
        """Calculate fuzzy match accuracy (>80% similarity)."""
        if not ground_truth:
            return 0.0

        matches = 0
        total = 0

        for key in ["merchant_name", "date"]:
            if key in ground_truth:
                total += 1
                similarity = fuzz.ratio(
                    str(extracted.get(key, "")),
                    str(ground_truth[key])
                ) / 100.0
                if similarity >= 0.8:
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _calculate_char_error_rate(self, extracted: str, ground_truth: str) -> float:
        """Calculate character-level error rate (Levenshtein distance)."""
        if not ground_truth:
            return 0.0

        from Levenshtein import distance
        
        max_len = max(len(extracted), len(ground_truth))
        if max_len == 0:
            return 0.0

        return distance(extracted, ground_truth) / max_len

    def _calculate_word_error_rate(self, extracted: str, ground_truth: str) -> float:
        """Calculate word-level error rate."""
        if not ground_truth:
            return 0.0

        extracted_words = extracted.split()
        truth_words = ground_truth.split()

        matches = sum(
            1 for e, t in zip(extracted_words, truth_words)
            if fuzz.ratio(e, t) >= 80
        )

        return 1.0 - (matches / max(len(truth_words), 1))

    def _calculate_field_completeness(self, extracted: Dict) -> float:
        """Calculate percentage of expected fields extracted."""
        expected_fields = {
            "merchant_name", "date", "total_amount", "items"
        }
        
        found_fields = {
            k for k in expected_fields
            if k in extracted and extracted[k]
        }
        
        return len(found_fields) / len(expected_fields)

    def _calculate_numerical_accuracy(self, extracted: Dict, ground_truth: Dict) -> float:
        """Calculate accuracy for numerical fields."""
        if "total_amount" not in ground_truth:
            return 0.0

        try:
            extracted_amount = float(extracted.get("total_amount", 0))
            truth_amount = float(ground_truth["total_amount"])
            
            if truth_amount == 0:
                return 1.0 if extracted_amount == 0 else 0.0
            
            error = abs(extracted_amount - truth_amount) / truth_amount
            # Allow 1% tolerance
            return max(0, 1.0 - error) if error <= 0.01 else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _calculate_consistency(self, extracted: Dict) -> float:
        """Validate against business rules."""
        score = 1.0

        # Check if items sum matches total
        try:
            items_total = sum(
                float(item.get("total", item.get("price", 0)))
                for item in extracted.get("items", [])
            )
            receipt_total = float(extracted.get("total_amount", 0))
            
            if receipt_total > 0 and abs(items_total - receipt_total) > 0.01:
                score -= 0.5
        except (ValueError, TypeError):
            score -= 0.3

        # Check date format
        if extracted.get("date"):
            try:
                from datetime import datetime
                datetime.strptime(extracted["date"], "%Y-%m-%d")
            except ValueError:
                score -= 0.2

        return max(0, score)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate benchmarking summary."""
        summary = {
            "total_tests": len(self.results),
            "timestamp": datetime.now().isoformat(),
            "providers": {},
        }

        for provider in self.extractors.keys():
            provider_metrics = [
                m for m in self.metrics
                if m.provider == provider
            ]

            if provider_metrics:
                summary["providers"][provider.value] = {
                    "count": len(provider_metrics),
                    "avg_field_accuracy": np.mean([m.field_accuracy for m in provider_metrics]),
                    "avg_fuzzy_accuracy": np.mean([m.fuzzy_accuracy for m in provider_metrics]),
                    "avg_char_error_rate": np.mean([m.char_error_rate for m in provider_metrics]),
                    "avg_word_error_rate": np.mean([m.word_error_rate for m in provider_metrics]),
                    "avg_processing_time": np.mean([m.processing_time for m in provider_metrics]),
                    "total_cost": sum([m.cost_per_page or 0 for m in provider_metrics]),
                    "avg_field_completeness": np.mean([m.field_completeness for m in provider_metrics]),
                    "avg_numerical_accuracy": np.mean([m.numerical_accuracy for m in provider_metrics]),
                    "avg_consistency_score": np.mean([m.consistency_score for m in provider_metrics]),
                }

        return summary

    def _save_results(self, output_dir: Path, summary: Dict[str, Any]):
        """Save benchmarking results."""
        # Save summary
        with open(output_dir / "benchmark_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed results
        with open(output_dir / "ocr_results.jsonl", "w") as f:
            for result in self.results:
                f.write(json.dumps(asdict(result), default=str) + "\n")

        # Save metrics
        with open(output_dir / "metrics.jsonl", "w") as f:
            for metric in self.metrics:
                f.write(json.dumps(asdict(metric), default=str) + "\n")

        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Initialize benchmark
    benchmark = OCRBenchmark(ground_truth_dir=Path("benchmarking/ground_truth"))

    # Register extractors
    benchmark.register_extractor(
        OCRProvider.GOOGLE_VISION,
        GoogleVisionExtractor()
    )
    
    benchmark.register_extractor(
        OCRProvider.GPT4O_MINI,
        GPT4oMiniExtractor(api_key=os.getenv("OPENAI_API_KEY"))
    )
    
    benchmark.register_extractor(
        OCRProvider.DEEPSEEK_R1,
        DeepSeekR1Extractor()
    )

    # Run benchmark
    results = benchmark.run_benchmark(
        image_dir=Path("benchmarking/test_receipts"),
        output_dir=Path("benchmarking/results")
    )

    # Print summary
    print(json.dumps(results, indent=2))
