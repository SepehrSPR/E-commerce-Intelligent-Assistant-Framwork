# Persian Query Handling System

A Persian query processing system with **multi-level classification** and **professional response generation** using **RAG (Retrieval-Augmented Generation)**.

Supports **Persian and English queries** with modular design, easy extensibility, and structured search argument extraction.

---

## Features

### Multi-Level Classification

- **Classifier A (`FaBERTClassifier`)**: Classifies queries into four main categories:
  - Site Support (`پشتیبانی سایت`)
  - Product Guidance (`راهنمایی در مورد کالا`)
  - Search (`جستجو`)
  - Comparison (`مقایسه`)

- **Classifier B (`FaBERTSubClassifier`)**: Further classifies search queries into:
  - Simple (`ساده`)
  - Parametric (`پارامتری`)
  - Non-Parametric (`غیرپارامتری`)

### RAG-Based Response Generation

- `SiteSupportRAG` → Handles site support questions  
- `ProductInfoRAG` → Provides product guidance answers  
- `ComparisonRAG` → Extracts two items for comparison queries  
- `ParameterizedRAG` → Extracts structured arguments for search queries using BM25, semantic search, and embeddings  

### Extensibility & Flexibility

- Replace classifiers with any Persian or multilingual model  
- Swap RAG LLMs with OpenAI, Cohere, DeepSeek, etc.  
- Fully customizable prompts for output style, tone, or format  
- Dataset classes and fields can be added, removed, or modified (adjust corresponding code modules and prompts)  
- Minor adjustments allow processing **English queries**  

---

## Project Structure

persian-query-system/
├── main.py
├── route.py
├── Classifier1.py
├── Classifier2.py
├── SiteSupportRag.py
├── Product_InfoDirectAsk.py
├── ComparisonRag.py
├── APIArgsFillerRag.py
├── datasets/
│ ├── classifier_a/
│ │ └── classifier_a.jsonl
│ ├── classifier_b/
│ │ └── classifier_b.jsonl
│ ├── site_support/
│ │ └── site_support.jsonl
│ ├── product_info/
│ │ └── product_info.jsonl
│ ├── comparison/
│ │ └── comparison.jsonl
│ └── search_queries/
│ ├── simple.jsonl
│ ├── parametric.jsonl
│ └── non_parametric.jsonl
├── models/
│ ├── FaBERTClassifier/
│ ├── FaBERTSubClassifier/
│ └── BGE/
└── requirements.txt

---

## Dataset Formats

### Classifier A (`classifier_a.jsonl`)
```json
{"query": "رمز عبورم را فراموش کرده‌ام", "label": "پشتیبانی سایت"}
{"query": "این گوشی چه ویژگی‌هایی دارد؟", "label": "راهنمایی در مورد کالا"}
{"query": "می‌خواهم بهترین لپ‌تاپ را پیدا کنم", "label": "جستجو"}
{"query": "مقایسه سامسونگ a71 و s23", "label": "مقایسه"}

### Classifier B (`classifier_b.jsonl`)
```json
{"query": "می‌خواهم بهترین لپ‌تاپ را پیدا کنم", "label": "ساده"}
{"query": "لپ‌تاپ با پردازنده i7 و رم ۱۶ گیگ", "label": "پارامتری"}
{"query": "لپ‌تاپ ارزان و با کیفیت", "label": "غیرپارامتری"}

### RAG Datasets Examples
### Site Support (site_support.jsonl)
```json
{"query": "چگونه حساب کاربری خود را فعال کنم؟", "answer": "برای فعال کردن حساب کاربری، روی لینک فعال‌سازی ایمیل کلیک کنید."}

### Product Info (product_info.jsonl)
```json
{"query": "این گوشی چه ویژگی‌هایی دارد؟", "answer": "گوشی X دارای ۸ گیگ رم و دوربین ۱۲ مگاپیکسل است."}

### Comparison (comparison.jsonl)
```json
{"query": "مقایسه گوشی سامسونگ a71 و s23", "item1": "سامسونگ a71", "item2": "سامسونگ s23"}
### Parametric Search Example (parametric.jsonl)
```json
{
  "query": "بهترین لپ‌تاپ برای برنامه نویسی",
  "queryTerms": "لپ‌تاپ، برنامه نویسی",
  "category": "لپ‌تاپ‌ها",
  "brand": "دل",
  "min_price": 20000000,
  "max_price": 30000000,
  "sort_order": "محبوب‌ترین"
}



