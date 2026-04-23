import os

import numpy as np
import requests
from chromadb.api.types import EmbeddingFunction


class HuggingFaceEmbeddingModel(EmbeddingFunction):
    """Using Hugging Face Embedding API with ChromaDB"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = self.get_hf_token()

        self.url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{model_name}/pipeline/feature-extraction"
        )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def get_hf_token():
        return os.getenv("HF_TOKEN")

    def __call__(self, texts: list[str]) -> list[np.array]:  # type: ignore
        payload = {"inputs": texts}

        r = requests.post(self.url, headers=self.headers, json=payload)

        if r.status_code != 200:
            raise RuntimeError(f"HuggingFace API error {r.status_code}: {r.text}")

        # Return embeddings data as float32 numpy arrays (what Chroma expects)
        data_json = r.json()
        data_array = [np.array(vec, dtype=np.float32) for vec in data_json]

        return data_array
