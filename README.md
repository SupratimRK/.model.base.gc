---
language:
- en
license: mit
library_name: onnx
pipeline_tag: text-classification
tags:
- gender-classification
- name-classification
- onnx
- quantized
- fast-inference
- text-classification
- scratch
- low-latency
metrics:
- accuracy
- f1
model-index:
- name: CompactGenderClassifier
  results:
  - task:
      type: text-classification
      name: Binary Gender Classification
    dataset:
      name: Curated Human First Names
      type: custom
    metrics:
    - name: F1 Score
      type: f1
      value: 0.8538
    - name: Accuracy
      type: accuracy
      value: 0.8538
widget:
- text: "Supratim Mondal"
  example_title: "Male Name"
- text: "Sayantika Biswas"
  example_title: "Female Name"
- text: "Alexander Hamilton"
  example_title: "Western Male Name"
- text: "Priya Sharma"
  example_title: "Female Name"
- text: "John Smith"
  example_title: "Western Male Name"
- text: "Emily Watson"
  example_title: "Western Female Name"
---

# CompactGenderClassifier: Ultra-Efficient Gender Classifier Trained From Scratch

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow.svg)](https://huggingface.co/SupratimRK/compact-gender-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/SupratimRK/.model.base.gc)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-INT8%20Quantized-blue)](https://onnxruntime.ai/)

**[🤗 Hugging Face Model Hub](https://huggingface.co/SupratimRK/compact-gender-classifier)** | **[🐙 GitHub Repository](https://github.com/SupratimRK/.model.base.gc)**

</div>

`CompactGenderClassifier` is a state-of-the-art, ultra-lightweight neural text classifier trained **from scratch** on a curated dataset of **197,165 labeled unique first names** (101,645 Female + 95,520 Male across diverse global and Indian origins).

Unlike generic large language models or pretrained BERT variants, `CompactGenderClassifier` has **zero external pretrained dependencies**, weighs just **~632 KB** (INT8 ONNX), and delivers sub-millisecond CPU inference (**~0.50 ms per query**).

---

## ⚡ Performance Highlights

| Metric | Previous Pretrained Model | **CompactGenderClassifier (Scratch)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size (INT8 ONNX)** | 4.30 MB (2.45 MB pruned) | **~632 KB** | **74%–85% smaller** |
| **Total Parameters** | 4,380,000 params | **532,482 params** | **88% parameter reduction** |
| **Vocabulary Size** | 30,522 tokens | **4,000 tokens** | **87% smaller vocabulary** |
| **Sequence Length** | 16 tokens | **20 tokens** | **No truncation blind spots** |
| **CPU Latency (Single Thread)** | ~1.20 ms | **~0.50 ms** | **~2.5x faster inference** |
| **Throughput** | ~800 queries/sec | **> 2,000 queries/sec** | Single CPU core |
| **Validation Accuracy** | 84.79% | **85.38%** | Evaluated on held-out clean test split |
| **Validation F1-Score** | — | **85.38%** | Balanced precision & recall |

---

## 🧠 Architectural Innovations

1. **Strict First-Name Purity**:
   - Trained strictly on clean given/first names where gender morphology is genuinely expressed.
   - Surnames are never assigned artificial gender labels, preventing spurious correlations and family name bias.

2. **Custom Domain WordPiece Tokenizer (4,000 Vocab)**:
   - Configured with `Sequence([Whitespace(), Punctuation(), Digits(individual_digits=True)])`.
   - Isolates subword affixes (e.g. `-a`, `-i`, `-ya`, `-ini`, `-wati` for female, and `-an`, `-ar`, `-esh`, `-kumar` for male) that signal gender morphology across cultures.
   - Cased vocabulary retains capitalization signals while casing augmentation ensures case invariance.

3. **Factorized Embeddings**:
   - Projects token vocabulary through `Embedding(4000, 64) -> Linear(64, 128, bias=False)`.
   - Cuts embedding parameter footprint in half with zero loss in representational capacity.

4. **Pre-LN Transformer Encoder**:
   - 2-layer, 128-hidden, 4-attention heads with GELU activations and Pre-LayerNorm (`norm_first=True`) for fast, stable convergence.

5. **Masked Dual Pooling (MeanPool + MaxPool)**:
   - Concatenates Masked Mean Pooling (128) + Masked Max Pooling (128) into a 256-dimensional representation.
   - Max Pooling acts as a sensitive detector for gender-bearing subword morphemes.

6. **Dual Runtime (Node.js & Python)**:
   - Pure Node.js runtime using `onnxruntime-node` requiring zero Python installed in production.
   - Built-in given-name extraction: handles single names (`"Supratim"`), full names (`"Supratim Mondal"`), and honorifics (`"Dr. Rahul Verma"`).

---

## 📦 Files in this Repository

| File | Size | Description |
| :--- | :--- | :--- |
| `models/model_int8.onnx` | ~632 KB | Quantized INT8 ONNX model (**recommended for production**) |
| `models/model.onnx` | ~2.1 MB | Full precision (FP32) ONNX model |
| `models/model_weights.pt` | ~2.1 MB | PyTorch model checkpoint |
| `models/tokenizer/vocab.txt` | ~26 KB | WordPiece vocabulary (4,000 tokens) |
| `models/tokenizer/tokenizer.json`| ~84 KB | Fast Rust-based Hugging Face tokenizer configuration |
| `config/model_config.json` | < 1 KB | Model architecture metadata & best validation scores |
| `scripts/classifier.mjs` | ~5.8 KB | Zero-Python Node.js inference engine |
| `scripts/predict.py` | ~5.2 KB | Python CLI & batch inference tool |
| `scripts/test_benchmark.mjs` | ~3.3 KB | Node.js accuracy & latency benchmark suite |
| `scripts/train.py` | ~9.8 KB | PyTorch GPU/CPU training pipeline |
| `scripts/export_onnx.py` | ~4.5 KB | FP32 export & INT8 dynamic quantization script |

---

## 🚀 Quickstart: Node.js (Zero-Python Production)

### Step 1: Install `onnxruntime-node`
```bash
npm install onnxruntime-node
```

### Step 2: Run Inference
```javascript
import { CompactGenderClassifier } from './scripts/classifier.mjs';

const classifier = new CompactGenderClassifier();
await classifier.init();

// Single first name
console.log(await classifier.predict('Sayantika'));
// Output: { gender: 'FEMALE', confidence: 0.952, raw_probs: { male: 0.048, female: 0.952 } }

// Full name with surname
console.log(await classifier.predict('Supratim Mondal'));
// Output: { gender: 'MALE', confidence: 0.967, given_name: 'Supratim', surnames: ['Mondal'], raw_probs: { male: 0.967, female: 0.033 } }

// Full name with title
console.log(await classifier.predict('Dr. Alexander Hamilton'));
// Output: { gender: 'MALE', confidence: 0.972, given_name: 'Alexander', surnames: ['Hamilton'] }
```

---

## 🐍 Quickstart: Python

### 1. Download Model from Hugging Face Hub
```python
from huggingface_hub import hf_hub_download

repo_id = "SupratimRK/compact-gender-classifier"

# Download ONNX INT8 model and tokenizer
onnx_path = hf_hub_download(repo_id=repo_id, filename="model_int8.onnx")
tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
```

### 2. Run via CLI
```bash
# Interactive mode
python predict.py -i

# Batch query
python predict.py --text "Supratim Mondal" "Sayantika Biswas" "Alexander Hamilton" "Priya Sharma"
```

### 3. Run in Code
```python
from predict import GenderClassifier

clf = GenderClassifier(use_onnx=True)
results = clf.predict(["Supratim Mondal", "Sayantika Biswas"])

for r in results:
    print(f"{r['text']} -> {r['gender']} ({r['confidence']*100:.1f}%) [Given: {r['given_name']}]")
```

---

## 🧪 Run Benchmarks
```bash
npm test
# or
node scripts/test_benchmark.mjs
```
