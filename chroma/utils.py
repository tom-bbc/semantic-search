import os

import numpy as np
from chromadb.api.types import EmbeddingFunction
from huggingface_hub import InferenceClient
from tqdm import tqdm


class HuggingFaceEmbeddingModel(EmbeddingFunction):
    """Using Hugging Face Embedding API with ChromaDB"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.api_key = self.get_hf_token()

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key,
        )

    @staticmethod
    def get_hf_token():
        return os.getenv("HF_TOKEN")

    def __call__(self, texts: list[str]) -> list[np.array]:  # type: ignore
        # Run inference on embedding model in HF Hub
        data_array = []

        for text in tqdm(texts, desc="Generating embeddings"):
            embedding = self.client.feature_extraction(
                text,
                model=self.model_name,
            )
            data_array.append(embedding)

        return data_array
