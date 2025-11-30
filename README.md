# E-commerce Intelligent Assistant Framework
A modular and extensible **E-commerce Intelligent Assistant Framework** designed for advanced Persian query processing.  
This system uses multi-level intent classification (FaBERT-based) combined with **RAG (Retrieval-Augmented Generation)** pipelines to deliver accurate, context-aware, and domain-specific responses.  
All components—including classifiers, datasets, embedders, APIs, prompts, and RAG modules—are fully customizable, and the framework can be adapted for English queries with enough adjustments.

## Features

- **Multi-Level Classification**
  - **Classifier A (`FaBERTClassifier`)**: Classifies queries into four categories:
    - Site Support (`پشتیبانی سایت`)
    - Product Guidance (`راهنمایی در مورد کالا`)
    - Search (`جستجو`)
    - Comparison (`مقایسه`)
  - **Classifier B (`FaBERTSubClassifier`)**: Further classifies search queries into:
    - Simple (`ساده`)
    - Parametric (`پارامتری`)
    - Non-Parametric (`غیرپارامتری`)

- **RAG-Based Response Generation**
  - `SiteSupportRAG`: Handles site support questions
  - `ProductInfoRAG`: Provides product guidance answers
  - `ComparisonRAG`: Extracts two items for comparison queries
  - `ParameterizedRAG`: Extracts structured arguments for search queries using BM25, semantic search, and embeddings

### **Extensibility & Flexibility**

- **Classifiers are interchangeable**  
  You can replace FaBERT-based classifiers with any other Persian or multilingual models (e.g., mBERT, XLM-R, GPT-based API classifiers, etc.).

- **RAG LLM backends are swappable**  
  You can use different LLM APIs such as DeepSeek, OpenAI, Cohere, or any custom local model.  
  Only the LLM wrapper class needs to be updated.

- **Prompts are fully customizable**  
  You can change reasoning style, tone, safety rules, extraction logic, and output format in each RAG module as needed.

- **Dataset schemas are flexible**  
  Dataset classes and fields can be added, removed, or renamed.  
  Whenever schema changes, make sure to update:  
  - The related model training scripts  
  - The routing logic in `route.py`  
  - Prompt construction in each RAG module  
  - Weaviate schema (if applicable)

- **Embedding model can be replaced easily**  
  The system currently uses a local **BGE embedding model**, but you can swap it with:  
  SentenceTransformers models, OpenAI embeddings, Cohere embeddings, or any HuggingFace embedding model.

- **Supports multilingual and English adaptation**  
  With proper dataset adjustments and prompt tuning, the entire system can be switched to English or become fully bilingual **without modifying the core architecture**.
## Project & Dataset Structure
```text
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
│   ├── classifier_a/
│   │   └── classifier_a.jsonl
│   ├── classifier_b/
│   │   └── classifier_b.jsonl
│   ├── site_support/
│   │   └── site_support.jsonl
│   ├── product_info/
│   │   └── product_info.jsonl
│   ├── comparison/
│   │   └── comparison.jsonl
│   └── search_queries/
│       ├── simple.jsonl
│       ├── parametric.jsonl
│       └── non_parametric.jsonl
├── models/
│   ├── FaBERTClassifier/
│   ├── FaBERTSubClassifier/
│   └── BGE/
```
### Classifier A
```json
{"کوئری": "رمز عبورم را فراموش کرده‌ام", "دسته": "پشتیبانی سایت"}
{"کوئری": "گوشی A51 چه ویژگی‌هایی دارد؟", "دسته": "راهنمایی در مورد کالا"}
{"کوئری": "می‌خواهم بهترین لپ‌تاپ را پیدا کنم", "دسته": "جستجو"}
{"کوئری": "مقایسه سامسونگ a71 و s23", "دسته": "مقایسه"}
```
### Classifier B
```json
{"کوئری": "خرید گوشی َ redmi note 14s", "دسته": "ساده"}
{"کوئری": "لپ‌تاپ با پردازنده i7 و رم ۱۶ گیگ", "دسته": "پارامتری"}
{"کوئری": "لپ‌تاپ ارزان و با کیفیت", "دسته": "غیرپارامتری"}
```



