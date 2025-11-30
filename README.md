# Persian Query Handling System

A Persian query processing system with multi-level classification and professional response generation using RAG (Retrieval-Augmented Generation). The system is modular, extendable, and can be adapted for English queries with enough adjustments.

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

- **Extensibility & Flexibility**
  - Classifiers can be replaced with other Persian or multilingual models
  - RAG LLMs can use alternative APIs (DeepSeek, OpenAI, Cohere, etc.)
  - Prompts are fully customizable to modify response behavior, tone, or output format
  - Dataset classes and fields can be added, removed, or modified, but corresponding adjustments must be applied in code modules and prompt-building logic
  - Compatible with Persian and English queries

## Project & Dataset Structure
'''
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
'''

