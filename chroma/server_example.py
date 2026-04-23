import json

import chromadb
import polars
from utils import HuggingFaceEmbeddingModel

DATA_CSV_FILE = "../bbc_news_transcripts.csv"


if __name__ == "__main__":
    # Create ChromaDB client that accesses the database from the docker compose service
    host = "127.0.0.1"
    port = 3030
    client = chromadb.HttpClient(host, port)

    # Create empty collection in the database
    collection_name = "transcripts"
    collection = client.get_or_create_collection(collection_name)

    # Load transcript data from local CSV file
    transcripts = polars.read_csv(DATA_CSV_FILE)
    print(transcripts.head())

    # Embedding model
    emb_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddingModel(
        model_name=emb_model_name,
    )

    subset_size = 10

    if collection.count() < subset_size:
        # Generate embeddings
        transcripts_subset = transcripts.sample(subset_size)
        documents = transcripts_subset["body"].to_list()

        metadata = [
            {"subject": subject} for subject in transcripts_subset["subject"].to_list()
        ]

        embeddings = embedding_model(documents)

        # Add new embeddings to vector database
        collection.add(
            ids=[f"id-{i}" for i in range(subset_size)],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,  # type: ignore
        )

    total_elements = collection.count()
    print(f"\nTotal elements in vector database: {total_elements}", end="\n\n")

    # Query vector database
    query = "Prime Minister's response to Peter Mandleson"
    n_results = 1

    query_embedding = embedding_model([query])
    result = collection.query(query_embeddings=query_embedding, n_results=n_results)

    print(f"Query: '{query}'", end="\n\n")
    print(f"Result: {json.dumps(result, indent=4)}")
