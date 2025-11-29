import requests
import weaviate
import torch
from typing import List, Optional
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from langchain.schema import Document
from langchain_core.language_models import LLM


WEAVIATE_CLASS_TRAIN = "" 

def get_train_class_by_query_class(query_class: str) -> str:
    mapping = {
        "ساده": "APIArgsFillingSimpleQuerytrain",
        "پارامتری": "APIArgsFillingParametricQuerytrain",
        "غیر پارامتری": "APIArgsFillingNonParametricQuerytrain"
    }
    return mapping.get(query_class, WEAVIATE_CLASS_TRAIN)

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

class ParameterizedRAG:
    def __init__(self, class_name: str):
        self.client = weaviate.connect_to_local()
        self.embedding_model = LocalBGEEmbedding(BGE_MODEL_PATH)
        self.llm = DeepSeekLLM(api_key=DEEPSEEK_API_KEY)
        self.class_name = class_name
        self.collection_name = get_train_class_by_query_class(class_name)

        self.category_collection_name = "CategoryCodes"
        self.brand_collection_name = "BrandCodes"

    def bm25_search(self, client, query: str, collection_name: str, k=5):
        collection = client.collections.get(collection_name)
        results = collection.query.bm25(query=query, limit=k)
        docs = []
        for obj in results.objects:
            props = obj.properties
            docs.append(Document(
                page_content=props.get("query", ""),
                metadata={
                    "queryTerms": props.get("queryTerms", ""),
                    "category": props.get("category", ""),
                    "brand": props.get("brand", ""),
                    "min_price": props.get("min_price", -1),
                    "max_price": props.get("max_price", -1),
                    "sort_order": props.get("sort_order", ""),
                }
            ))
        return docs

    def semantic_search(self, client, query: str, embedding_model, collection_name: str, k=5):
        collection = client.collections.get(collection_name)
        vector = embedding_model.embed_query(query)
        results = collection.query.near_vector(near_vector=vector, limit=k)
        docs = []
        for obj in results.objects:
            props = obj.properties
            docs.append(Document(
                page_content=props.get("query", ""),
                metadata={
                    "queryTerms": props.get("queryTerms", ""),
                    "category": props.get("category", ""),
                    "brand": props.get("brand", ""),
                    "min_price": props.get("min_price", -1),
                    "max_price": props.get("max_price", -1),
                    "sort_order": props.get("sort_order", ""),
                }
            ))
        return docs

    
    def build_prompt(self, query: str, examples: List[Document]) -> str:
        prefix_map = {
            "ساده": "در این دسته از کوئری‌ها که ساده هستند، مدل دقیق کالا حتماً ذکر می‌شود.",
            "پارامتری": "در این دسته از کوئری‌ها که پارامتری هستند، مدل دقیق کالا ذکر نمی‌شود.",
            "غیر پارامتری": (
                "در این دسته از کوئری‌ها که غیر پارامتری هستند، کلمات کلی مثل ارزان‌ترین، خوش‌قیمت، بهترین، "
                "پرتخفیف‌ترین، پربازدیدترین و جملات و کلمات هم‌معنی حضور دارند که در کلمات کلیدی نباید آورده شوند.\n"
                )
        }
        prefix = prefix_map.get(self.class_name, "")
        prompt = (
            f"{prefix}\n"
            "وظیفه تو استخراج آرگومان‌ها از پرسش کاربر است.\n"
            "در ادامه چند نمونه کوئری به همراه آرگومان‌های استخراج‌شده از آن‌ها ارائه شده است:\n"
            "قالب پاسخ دقیقاً به صورت زیر است:\n"
            "کلمات کلیدی: مقدار، دسته: مقدار، برند: مقدار، حداقل قیمت: مقدار، حداکثر قیمت: مقدار، ترتیب: مقدار\n"
            "دقت کن:\n"
            "در کوئری‌هایی مثل «فلان کالا برای فلان چیز»، فلان کالا در تشخیص آرگومان‌ها مهم است و فلان چیز فقط در کلمات کلیدی می‌آید.\n"
            "تا زمانی که برند کالا صراحتاً ذکر نشده باشد، برند عمومی باقی می‌ماند.\n"
            "کلمات برند یا مدل نباید در کلمات کلیدی باشند.\n"
            "اگر برندها وکتگوری هایی که از مثال درآوردی با اینایی که بالا اول بهت دادم تختلاف املایی یا نوشتاری داشت به مناسب ترین بالایی ها تبدیلشون کن، مگر اینکه بر طبق الگو عمومی تشخیص دادی اونموقع دست نزن.\n"
        )

        for i, ex in enumerate(examples, 1):
            md = ex.metadata
            prompt += (
                f"\nنمونه‌ی {i}:\n"
                f"سوال: {ex.page_content}\n"
                f"پاسخ: کلمات کلیدی: {md.get('queryTerms', '')}، دسته: {md.get('category', '')}، "
                f"برند: {md.get('brand', '')}، حداقل قیمت: {md.get('min_price', -1)}، "
                f"حداکثر قیمت: {md.get('max_price', -1)}، ترتیب: {md.get('sort_order', '')}، "
                f"کلاس کوئری: {md.get('query_class', '')}، {self.class_name}\n"
            )

        prompt += (
            f"\nسوال کاربر:\n\"{query}\"\n"
            "آرگومان‌ها را با توجه به الگوی نمونه‌های بالا استخراج کن و فقط در همان قالب بنویس.\n"
            "مقدار هیچ‌کدام را خالی نگذار. هیچ توضیح، تفسیر، متن اضافه یا فرمت متفاوت ارائه نده.\n"
        )
        return prompt

    def generate_answer(self, query: str):
        bm25_docs = self.bm25_search(self.client, query, self.collection_name, 5)
        sem_docs = self.semantic_search(self.client, query, self.embedding_model, self.collection_name, 5)
        examples = bm25_docs + sem_docs
        prompt_main = self.build_prompt(query, examples)
        answer_main = self.llm._call(prompt_main)
        self.client.close()
        return answer_main



