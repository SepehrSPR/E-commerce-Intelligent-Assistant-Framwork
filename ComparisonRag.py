import requests
import weaviate
import torch
from typing import List, Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain_core.language_models import LLM

WEAVIATE_CLASS_NAME = "ComparisonTrain"
DEEPSEEK_API_KEY = "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"
BGE_MODEL_PATH = r"F:\Arshad\Payanname\BGE-m3 model"


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
            "model": "deepseek/deepseek-chat-v3-0324",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
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


class ComparisonRAG:
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
                metadata={
                    "item1": props.get("item1", ""),
                    "item2": props.get("item2", "")
                }
            ))
        return docs

    def semantic_search(self, client, query: str, embedding_model, collection_name: str, k=5):
        vec = embedding_model.embed_query(query)
        collection = client.collections.get(collection_name)
        results = collection.query.near_vector(near_vector=vec, limit=k)
        docs = []
        for obj in results.objects:
            props = obj.properties
            docs.append(Document(
                page_content=props.get("query", ""),
                metadata={
                    "item1": props.get("item1", ""),
                    "item2": props.get("item2", "")
                }
            ))
        return docs

    def build_prompt(self, user_query: str, examples: List[Document]) -> str:
        context = """شما یک مدل زبان هستید که دو آیتم مقایسه‌ای را از سوال کاربر استخراج می‌کنید.

    قالب خروجی:  
    آیتم اول: <نام آیتم اول استخراجی>، آیتم دوم: <نام آیتم دوم استخراجی>

    نکته ی اول :اگر در سوال فقط یک آیتم ذکر شده اولین آیتم استراجی بصورت جمع آورده بیاور، و آیتم دوم را "یکدیگر" قرار بده ضمن اینکه ذکر کلمه ی مدل یا مدل ها در نام آیتم استخراجی مجاز نمی باشد\n 
    نکته ی دوم : در صورت ذکر نام برند مثل سامسونگ a5, سامسونگ a50 یا نوع کالا مثل گوشی a51 یا هر دو مثل گوشی سامسونگ a71 در مقابل s23 در کوئری باید این موارد(برند و نوع کالا) در هر دوی آیتم های استخراجی آورده شود ولی اگر هر کدوم در کوئری نبود خودت آنرا اضافه نکن.\n
    نکته ی سوم: هیچ کلمه یا حرف فارسی رو از خودت انگلیسی نکن و بر عکس.\n
    ی استخراج شده را نیز جمع بیاور.

    نمونه‌ها:
    """
        for i, ex in enumerate(examples, 1):
            context += f"سوال: {ex.page_content}\n"
            context += f"پاسخ: آیتم اول: {ex.metadata.get('item1', '')}، آیتم دوم: {ex.metadata.get('item2', '')}\n\n"
        context += f"""---

    سوال جدید:  
    {user_query}

    لطفاً فقط جواب را دقیقاً در قالب بالا و بدون هیچ متن اضافی یا توضیح ارائه بده.
    """
        return context
    
    def generate_answer(self, query: str) -> str:
        bm25_docs = self.bm25_search(self.client, query, self.collection_name, 5)
        sem_docs = self.semantic_search(self.client, query, self.embedding_model, self.collection_name, 5)
        examples = bm25_docs + sem_docs
        prompt = self.build_prompt(query, examples)
        print(f"پرامپت داده شده به مدل :  \n{prompt}")
        self.client.close()
        return self.llm._call(prompt)
