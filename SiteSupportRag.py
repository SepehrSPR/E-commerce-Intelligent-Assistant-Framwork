import requests
import weaviate
import torch
from typing import List, Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain_core.language_models import LLM


WEAVIATE_CLASS_NAME = ...
DEEPSEEK_API_KEY = ...
BGE_MODEL_PATH = ...

class DeepSeekLLM(LLM, BaseModel):
    api_key: str = Field(..., exclude=True)

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": ...,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        response = requests.post(..., headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class LocalBGEEmbedding:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_query(self, query: str) -> List[float]:
        with torch.no_grad():
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().squeeze(0).tolist()


class SiteSupportRAG:
    def __init__(self):
        self.client = weaviate.connect_to_local()
        self.embedding_model = LocalBGEEmbedding(BGE_MODEL_PATH)
        self.llm = DeepSeekLLM(api_key=DEEPSEEK_API_KEY)
        self.collection_name = WEAVIATE_CLASS_NAME

    def bm25_search(self, client, query: str, collection_name: str, k=5):
        collection = client.collections.get(collection_name)
        results = collection.query.bm25(query=query, limit=k)
        docs = []
        for obj in results.objects:
            props = obj.properties
            docs.append(Document(
                page_content=props.get("query", ""),
                metadata={"answer": props.get("answer", "")}
            ))
        return docs

    def semantic_search(self, client, query, embedding_model, collection_name, k=5):
        query_vec = embedding_model.embed_query(query)
        collection = client.collections.get(collection_name)
        results = collection.query.near_vector(near_vector=query_vec, limit=k)
        docs = []
        for obj in results.objects:
            props = obj.properties
            docs.append(Document(
                page_content=props.get("query", ""),
                metadata={"answer": props.get("answer", "")}
            ))
        return docs

    def deduplicate_docs(self, docs: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)
        return unique_docs

    def build_prompt(self, user_query: str, examples: List[Document]) -> str:
        context = """تو یک ربات پاسخ دهی به سوالات پشتیبانی کاربران هستی.\n
        وظیفه ی تو بازگرداندن متن دقیق پاسخ یکی از مثال ها و یا در برخی موارد ترکیب دو یا چند پاسخ دقیق از مثال ها برای رسیدن به جواب صحیح است.\n
        در ادامه چند نمونه سوال و پاسخ آورده شده است:\n"""
        for i, ex in enumerate(examples, 1):
            context += f"\nنمونه {i}:\n"
            context += f"سوال: {ex.page_content}\n"
            context += f"پاسخ: {ex.metadata.get('answer', '')}\n"
        context += f"\nسوال جدید:\n{user_query}\n"
        context += "لطفاً پاسخ مناسب را با توجه به نمونه‌ها برای این سوال طبق دستورالعمل داده شده تولید کن ."
        return context

    def generate_answer(self, query: str) -> str:
        bm25_docs = self.bm25_search(self.client, query, self.collection_name, 5)
        sem_docs = self.semantic_search(self.client, query, self.embedding_model, self.collection_name, 5)
        examples = self.deduplicate_docs(bm25_docs + sem_docs)
        prompt = self.build_prompt(query, examples)
        self.client.close()
        return self.llm._call(prompt)
