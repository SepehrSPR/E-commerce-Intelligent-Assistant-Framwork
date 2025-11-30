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
  You can change reasoning style, extraction logic, and output format in each RAG module as needed.

- **Dataset schemas are flexible**  
  Dataset classes and fields can be added, removed, or renamed.  
  Whenever schema changes, make sure to update:  
  - The related model training scripts  
  - The routing logic in `route.py`  
  - Prompt construction in each RAG module  
  - Weaviate schema

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
├── ProductTechnicalSupportDirectAsk.py
├── ComparisonRag.py
├── APIArgsFillerRag.py
├── datasets/
│   ├── classifier_a/
│   │   └── classifier_a.jsonl
│   ├── classifier_b/
│   │   └── classifier_b.jsonl
│   ├── site_support/
│   │   └── site_support.jsonl
│   ├── comparison/
│   │   └── comparison.jsonl
│   └── APIArgsFiller/
│       ├── simple.jsonl
│       ├── parametric.jsonl
│       └── non_parametric.jsonl
```
### Classifier A dataset example
```json
{"کوئری": "رمز عبورم را فراموش کرده‌ام", "دسته": "پشتیبانی سایت"}
{"کوئری": "گوشی A51 چه ویژگی‌هایی دارد؟", "دسته": "راهنمایی در مورد کالا"}
{"کوئری": "می‌خواهم بهترین لپ‌تاپ را پیدا کنم", "دسته": "جستجو"}
{"کوئری": "مقایسه سامسونگ a71 و s23", "دسته": "مقایسه"}
```
### Classifier B dataset example
```json
{"کوئری": "خرید گوشی َ redmi note 14s", "دسته": "ساده"}
{"کوئری": "لپ‌تاپ با پردازنده i7 و رم ۱۶ گیگ", "دسته": "پارامتری"}
{"کوئری": "لپ‌تاپ ارزان و با کیفیت", "دسته": "غیرپارامتری"}
```
### Site Support dataset example
```json
{"کوئری": "چگونه حساب کاربری خود را فعال کنم؟", "پاسخ": "برای فعال کردن حساب کاربری، روی لینک فعال‌سازی ایمیل کلیک کنید."}
```
### Comparison dataset example
```json
{"query": "مقایسه گوشی سامسونگ a71 و s23", "item1": "سامسونگ a71", "item2": "سامسونگ s23"}
```
### APIArgsFiller datasets example
**simple dataset
```json
"query":"مانیتور ال‌جی 27GP850-B با صفحه بزرگ و نرخ تازه‌سازی بالا می‌خوام","queryTerms":"[مانیتور,ال‌جی,27GP850-B,با,صفحه,بزرگ,نرخ,تازه‌سازی,بالا]","CategoryFilter":"category:مانیتور","BrandFilter":"brand:ال‌جی","minPrice":"0","maxPrice":"0","sortOrder":"normal"```
```
**Parametric dataset
```json
"query":"ساعت هوشمند مدل جدید سامسونگ، در حد نو، بدون خط و خش، قیمت حدود ۶ میلیون تومان.","queryTerms":"[ساعت,هوشمند جدید,سامسونگ,در,حد,نو,بدون,خط,خش]","CategoryFilter":"category:ساعت و بند هوشمند","BrandFilter":"brand:سامسونگ","minPrice":"6000000","maxPrice":"6000000","sortOrder":"normal"
```






