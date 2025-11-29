# Persian Query Handling System

A Persian query processing system with multi-level classification and professional response generation using RAG (Retrieval-Augmented Generation).

The system supports:
- Multi-level query classification
- Argument extraction for structured search queries
- Professional answers for site support, product information, comparison, and search queries
- Modular, extendable design compatible with Persian and English queries

---

## Features

- **Multi-Level Classification**
  - **Classifier A (`FaBERTClassifier`)**: Classifies queries into four main categories:
    - Site Support (`پشتیبانی سایت`)
    - Product Guidance (`راهنمایی در مورد کالا`)
    - Search (`جستجو`)
    - Comparison (`مقایسه`)
  - **Classifier B (`FaBERTSubClassifier`)**: Further classifies search queries into:
    - Simple (`ساده`)
    - Parametric (`پارامتری`)
    - Non-Parametric (`غیرپارامتری`)

- **RAG-Based Response Generation**
  - `SiteSupportRAG` → Handles site support questions
  - `ProductInfoRAG` → Provides product guidance answers
  - `ComparisonRAG` → Extracts two items for comparison queries
  - `ParameterizedRAG` → Extracts structured arguments for search queries using BM25, semantic search, and embeddings

- **Extensibility & Flexibility**
  - Classifiers can be replaced with any other Persian or multilingual models
  - RAG LLMs can use alternative APIs such as OpenAI, Cohere, or DeepSeek
  - Prompts are fully customizable to modify response behavior, tone, or output format
  - Dataset classes and fields can be **added, removed, or modified**, but corresponding adjustments **must** be applied in related code modules and prompt-building logic
  - With minor adjustments, the system can also handle **English queries**

---

## Project & Dataset Structure

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
│ ├── classifier_a/ # Training dataset for Classifier A
│ │ └── classifier_a.jsonl
│ ├── classifier_b/ # Training dataset for Classifier B
│ │ └── classifier_b.jsonl
│ ├── site_support/ # Site support dataset
│ │ └── site_support.jsonl
│ ├── product_info/ # Product info dataset
│ │ └── product_info.jsonl
│ ├── comparison/ # Comparison dataset
│ │ └── comparison.jsonl
│ └── search_queries/ # Search queries: Simple / Parametric / Non-Parametric
│ ├── simple.jsonl
│ ├── parametric.jsonl
│ └── non_parametric.jsonl
├── models/ # FaBERT and embedding models
│ ├── FaBERTClassifier/
│ ├── FaBERTSubClassifier/
│ └── BGE/
└── requirements.txt

yaml
Copy code

---

## Dataset Formats for Classifier Training

### Classifier A (`classifier_a.jsonl`)

```json
{"query": "رمز عبورم را فراموش کرده‌ام", "label": "پشتیبانی سایت"}
{"query": "این گوشی چه ویژگی‌هایی دارد؟", "label": "راهنمایی در مورد کالا"}
{"query": "می‌خواهم بهترین لپ‌تاپ را پیدا کنم", "label": "جستجو"}
{"query": "مقایسه سامسونگ a71 و s23", "label": "مقایسه"}
Classifier B (classifier_b.jsonl)
json
Copy code
{"query": "می‌خواهم بهترین لپ‌تاپ را پیدا کنم", "label": "ساده"}
{"query": "لپ‌تاپ با پردازنده i7 و رم ۱۶ گیگ", "label": "پارامتری"}
{"query": "لپ‌تاپ ارزان و با کیفیت", "label": "غیر پارامتری"}
Each line is a separate JSON object containing the query text and its label. Labels and query text can be adapted for English queries. Dataset classes and fields can be expanded or modified, but corresponding adjustments must be applied in code modules, embeddings, and prompts.

Other RAG Datasets
1. Site Support (site_support.jsonl)
json
Copy code
{"query": "چگونه حساب کاربری خود را فعال کنم؟", "answer": "برای فعال کردن حساب کاربری، روی لینک فعال‌سازی ایمیل کلیک کنید."}
2. Product Info (product_info.jsonl)
json
Copy code
{"query": "این گوشی چه ویژگی‌هایی دارد؟", "answer": "گوشی X دارای ۸ گیگ رم و دوربین ۱۲ مگاپیکسل است."}
3. Comparison (comparison.jsonl)
json
Copy code
{"query": "مقایسه گوشی سامسونگ a71 و s23", "item1": "سامسونگ a71", "item2": "سامسونگ s23"}
4. Search Queries (simple.jsonl, parametric.jsonl, non_parametric.jsonl)
json
Copy code
{
  "query": "بهترین لپ‌تاپ برای برنامه نویسی",
  "queryTerms": "لپ‌تاپ، برنامه نویسی",
  "category": "لپ‌تاپ‌ها",
  "brand": "دل",
  "min_price": 20000000,
  "max_price": 30000000,
  "sort_order": "محبوب‌ترین"
}
Model Training & Preparation
FaBERT Classifiers

Train using datasets/classifier_a/ and datasets/classifier_b/

Use transformers library (AutoModelForSequenceClassification)

Save models under models/FaBERTClassifier and models/FaBERTSubClassifier

Embedding & RAG

Use BGE model in models/BGE for semantic embeddings

Import datasets into Weaviate collections based on their type

Flexibility

Classifier models can be replaced with other models (Persian or English)

LLM APIs for RAG can be swapped (DeepSeek, OpenAI, Cohere, etc.)

Dataset classes and fields can be expanded or modified, with required code adjustments

Prompt Customization

Adjust build_prompt functions in each RAG module

Modify output format, tone, or extraction logic

Running the System
bash
Copy code
python main.py
Queries are routed through classifiers

Appropriate RAG module generates the response

Exit anytime using exit or خروج

Installation & Dependencies
bash
Copy code
pip install torch transformers langchain weaviate-client pydantic requests
Ensure API keys (DEEPSEEK_API_KEY) and model paths (BGE_MODEL_PATH) are set correctly. GPU is recommended for embeddings and inference.

Usage Examples (Persian queries)
Product Information
python
Copy code
from Product_InfoDirectAsk import ProductInfoRAG

rag = ProductInfoRAG()
query = "این لپ‌تاپ چه مشخصاتی دارد؟"
answer = rag.generate_answer(query)
print(answer)
Comparison Queries
python
Copy code
from ComparisonRag import ComparisonRAG

rag = ComparisonRAG()
query = "مقایسه گوشی سامسونگ a71 و s23"
answer = rag.generate_answer(query)
print(answer)
Parameterized Search
python
Copy code
from APIArgsFillerRag import ParameterizedRAG

rag = ParameterizedRAG(class_name="پارامتری")
query = "می‌خواهم بهترین لپ‌تاپ با بودجه ۳۰ میلیون پیدا کنم"
answer = rag.generate_answer(query)
print(answer)
With minor adjustments to dataset and prompts, the system can process English queries effectively.

Extensibility & Customization
Replace classifiers or embeddings with alternative models

Use different LLM APIs for RAG modules

Adjust prompts to customize output format, tone, and extraction logic

Dataset classes and fields can be added, removed, or modified, but corresponding adjustments must be applied in related code modules and prompt-building logic

Compatible with Persian and English queries (or other languages with necessary adaptations)

License
MIT License © 2025

yaml
Copy code

---







