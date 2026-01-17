# OCR Benchmarking Suite

Comprehensive benchmarking system comparing Google Vision API, GPT-4o mini, and DeepSeek R1 for receipt OCR extraction.

## Overview

This system evaluates OCR providers across multiple dimensions:

- **Accuracy**: Field-level exact/fuzzy matching, character/word error rates
- **Performance**: Processing time per document
- **Cost**: API costs vs. local inference
- **Quality**: Field completeness, numerical accuracy, business logic validation

## Architecture

```
benchmarking/
├── ocr_benchmark_engine.py      # Core benchmarking engine
├── run_benchmark.py              # CLI runner with reporting
├── .env.example                  # Environment variables template
├── requirements.txt              # Python dependencies
├── test_receipts/                # Test receipt images
├── ground_truth/                 # Ground truth JSON annotations
└── results/                      # Output results and reports
```

## Quick Start

### 1. Install Dependencies

```bash
cd benchmarking
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
export OPENAI_API_KEY="your-key-here"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/gcp-key.json"
```

### 3. Prepare Test Data

#### Add Test Receipts

Place receipt images in `benchmarking/test_receipts/`:

```bash
# Example structure
benchmarking/test_receipts/
├── receipt_001.png
├── receipt_002.jpg
└── ...
```

#### Create Ground Truth Annotations

For each receipt image, create a JSON file in `benchmarking/ground_truth/`:

```bash
# receipt_001.json
{
  "merchant_name": "Tesco Express",
  "date": "2024-01-15",
  "time": "14:30",
  "total_amount": 12.45,
  "tax_amount": 2.08,
  "items": [
    {
      "description": "Milk 1L",
      "quantity": 1,
      "unit_price": 1.20,
      "total": 1.20
    },
    {
      "description": "Bread",
      "quantity": 1,
      "unit_price": 1.50,
      "total": 1.50
    }
  ],
  "payment_method": "card",
  "raw_text": "[full OCR text from receipt]"
}
```

### 4. Run Benchmarks

#### Basic run (GPT-4o mini + DeepSeek R1)

```bash
python run_benchmark.py \
  --image-dir test_receipts \
  --ground-truth-dir ground_truth \
  --output-dir results
```

#### Include Google Vision

```bash
python run_benchmark.py \
  --image-dir test_receipts \
  --providers google_vision gpt4o_mini deepseek_r1 \
  --output-dir results
```

#### Skip visualization (for headless environments)

```bash
python run_benchmark.py \
  --skip-visualization \
  --output-dir results
```

### 5. Review Results

After running, check outputs in `benchmarking/results/`:

```
results/
├── benchmark_summary.json        # Aggregated metrics
├── benchmark_report.txt          # Human-readable report
├── benchmark_comparison.png      # Visualization charts
├── ocr_results.jsonl             # Raw OCR extraction results
└── metrics.jsonl                 # Detailed metrics per receipt
```

## Understanding Metrics

### Accuracy Metrics

- **Field Accuracy**: Exact match on key fields (merchant, date, total)
- **Fuzzy Accuracy**: >80% similarity matching
- **Character Error Rate (CER)**: Levenshtein distance / max length
- **Word Error Rate (WER)**: Word-level matching failures

### Performance Metrics

- **Processing Time**: Seconds per receipt
- **Tokens Used**: LLM token consumption (for GPT-4o mini, DeepSeek)
- **Cost Per Page**: Estimated API or compute cost

### Quality Metrics

- **Field Completeness**: % of expected fields found
- **Numerical Accuracy**: ±1% tolerance for amounts
- **Consistency Score**: Validation against business rules:
  - Items total matches receipt total (±0.01)
  - Date in valid format
  - All amounts positive

## DeepSeek R1 Optimization Strategy

Use GPT-4o mini results as the benchmark, then optimize DeepSeek R1:

### 1. Prompt Engineering

```python
# Use same structured prompt as GPT-4o mini
prompt = """
Extract receipt data and return as JSON:
{
  "merchant_name": "store name",
  "date": "YYYY-MM-DD",
  "total_amount": 0.00,
  "items": [{"description": "...", "quantity": 1, "price": 0.00}]
}
Return ONLY valid JSON.
"""
```

### 2. Local Deployment (Ollama/vLLM)

```bash
# Using Ollama
ollama pull deepseek-r1
ollama serve

# Then in Python
from ocr_benchmark_engine import DeepSeekR1Extractor
extractor = DeepSeekR1Extractor()
```

### 3. Fuzzy Matching Post-Processing

```python
# For close matches (80%+ similarity), accept as correct
if fuzz.ratio(extracted, ground_truth) >= 80:
    # Mark as acceptable
```

### 4. Batch Processing

```bash
# For high throughput
python run_benchmark.py \
  --providers deepseek_r1 \
  --batch-size 32
```

### 5. GPU Optimization

```bash
# Use vLLM for optimal inference
export CUDA_VISIBLE_DEVICES=0
python run_benchmark.py \
  --enable-vllm \
  --tensor-parallel 2
```

## Interpretation Guide

### Expected Performance Comparison

| Metric | Google Vision | GPT-4o mini | DeepSeek R1 |
|--------|---------------|-------------|-------------|
| **Field Accuracy** | 85-90% | 92-96% | 88-94% (goal) |
| **Processing Time** | 1.5-3s | 2-4s | 0.5-1.5s (local) |
| **Cost/Receipt** | $0.0015 | $0.001-0.002 | ~$0.00001 (local) |
| **Speed Advantage** | Baseline | Higher accuracy | Lowest cost |

### Key Findings from 2025 Research [web:17][web:18][web:19]

1. **GPT-4o mini outperforms Google Vision** in accuracy for complex receipts
2. **DeepSeek-OCR competitive speed** at 150 pages/min vs Google's 120
3. **Local deployment eliminates API costs** - critical for scale
4. **Multimodal understanding** matters for handwritten/mixed content

## Advanced Usage

### Custom Metrics

Extend `BenchmarkMetrics` class in `ocr_benchmark_engine.py`:

```python
class MyCustomMetrics(BenchmarkMetrics):
    def calculate_line_item_accuracy(self):
        # Your custom logic
        pass
```

### Integration with CI/CD

```yaml
# GitHub Actions example
- name: Run OCR Benchmarks
  run: |
    cd benchmarking
    python run_benchmark.py \
      --providers deepseek_r1 gpt4o_mini \
      --output-dir results
    
- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: benchmark-results
    path: benchmarking/results/
```

### Batch Testing

```python
# Process large receipt datasets
from pathlib import Path
import subprocess

for batch_dir in Path("receipts").glob("batch_*"):
    subprocess.run([
        "python", "run_benchmark.py",
        "--image-dir", str(batch_dir),
        "--output-dir", f"results/{batch_dir.name}"
    ])
```

## Troubleshooting

### Google Vision API

```bash
# Verify credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/key.json"
python -c "from google.cloud import vision; print('OK')"
```

### OpenAI/GPT-4o mini

```bash
# Test API key
export OPENAI_API_KEY="sk-..."
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### DeepSeek R1

```bash
# Test Ollama setup
ollama list  # Should show deepseek-r1
ollama pull deepseek-r1

# Or test vLLM
python -c "from vllm import LLM; LLM(model='deepseek-ai/DeepSeek-OCR')"
```

## Performance Benchmarks

Expected throughput on RTX 4090 (local):

- **DeepSeek R1**: ~3-5 pages/sec
- **GPT-4o mini**: Limited by API rate (100 req/min)
- **Google Vision**: Limited by API rate (1800 pages/min)

## References

1. [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR) [web:29]
2. [DeepSeek OCR vs Google Vision Speed Test](https://sparkco.ai/blog/deepseek-ocr-vs-google-vision-speed-test-analysis) [web:17]
3. [OCR Models Comparative Analysis](https://openaccess.thecvf.com/content/WACV2025W/VISIONDOCS) [web:19]
4. [Invoice Information Extraction Methods](https://arxiv.org/html/2510.15727v1) [web:21]
5. [Receipt Data Extraction Best Practices](https://skywork.ai/blog/ai-agent/deepseek-ocr-use-cases-2025/) [web:20]

## Contributing

To improve the benchmarking suite:

1. Add new metrics to `BenchmarkMetrics`
2. Extend extractors with new providers
3. Submit pull requests with test data

## License

Same as parent repository (models-ocr-pepline)
