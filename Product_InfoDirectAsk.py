import requests
from pydantic import BaseModel, Field
from langchain_core.language_models import LLM
from typing import List, Optional

DEEPSEEK_API_KEY = "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"

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


class ProductInfoRAG:
    def __init__(self):
        self.llm = DeepSeekLLM(api_key=DEEPSEEK_API_KEY)

    def generate_answer(self, query: str) -> str:
        prompt = f".تو یک دستیار پاسخگو به سوالات کاربران درباره ی مسائل فنی محصولات مختلف هستی لطفا مودبانه و با احترام به این سوال کاریر پاسخ بده\nسوال : {query}\n"
        return self.llm._call(prompt)
