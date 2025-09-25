from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

if __name__ == "__main__":
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:v1.5",
        base_url="http://localhost:11434",
        embed_batch_size=1,
    )
    text = "Who is the Mayor?"
    embedding = embed_model.get_general_text_embedding(text)
    print(embedding)
