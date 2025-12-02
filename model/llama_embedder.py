# llama_embedder.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


class LlamaEmbedder:
    def __init__(self, provider="hf", model_name=None, openai_key=None):
        if provider == "hf":
            if model_name is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.embedder = HuggingFaceEmbedding(model_name=model_name)

        elif provider == "openai":
            if openai_key is None:
                raise ValueError("OpenAI API key is required")
            self.embedder = OpenAIEmbedding(api_key=openai_key, model_name=model_name)

        else:
            raise ValueError("Unknown embedder provider")

    def embed(self, text: str):
        return self.embedder.get_text_embedding(text)
