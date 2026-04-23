import json
from uuid import uuid4

import chromadb
import polars
from utils import HuggingFaceEmbeddingModel

DATA_CSV_FILE = "../bbc_news_transcripts.csv"


def preprocess_dataframe(dataframe):

    new_dataframe = (
        dataframe
        # 1) Add a unique ID per original row
        .with_row_index("row_id")
        # 2) Split the target column on the substring delimiter
        .with_columns(polars.col("body").str.split("NEW STORY").alias("new_story"))
        # 3) Turn list of chunks into multiple rows (duplicating other columns)
        .explode("new_story")
        # 4) Clean + remove empties
        .with_columns(polars.col("new_story").str.strip_chars()).filter(
            polars.col("new_story").is_not_null() & (polars.col("new_story") != "")
        )
    )

    return new_dataframe


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
    transcripts = preprocess_dataframe(transcripts)
    print(transcripts.head())

    # Embedding model
    emb_model_name = "microsoft/harrier-oss-v1-270m"
    # emb_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_model = HuggingFaceEmbeddingModel(
        model_name=emb_model_name,
    )

    subset_size = 100

    if collection.count() < subset_size:
        # Generate embeddings
        transcripts_subset = transcripts.sample(subset_size)
        documents = transcripts_subset["new_story"].to_list()
        metadata = [
            {"subject": str(row["subject"]), "timestamp": str(row["date"])}
            for row in transcripts_subset.iter_rows(named=True)
        ]

        embeddings = embedding_model(documents)
        print(f"Embeddings: ({len(embeddings)}, {len(embeddings[0])})")

        # Add new embeddings to vector database
        collection.add(
            ids=[str(uuid4()) for _ in range(subset_size)],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,  # type: ignore
        )

    total_elements = collection.count()
    print(f"\nTotal elements in vector database: {total_elements}", end="\n\n")

    # Query vector database
    query = "Update on the war in the Middle East"
    n_results = 3

    query_embedding = embedding_model([query])
    result = collection.query(query_embeddings=query_embedding, n_results=n_results)

    print(f"Query: '{query}'", end="\n\n")
    print(f"Result: {json.dumps(result, indent=4)}")
