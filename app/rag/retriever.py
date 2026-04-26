import uuid
import chromadb
from app.rag.embeddings import get_embedding

chroma_client = chromadb.PersistentClient(path="chroma_db")

collection = chroma_client.get_or_create_collection(
    name="documents"
)


def save_chunks(chunks: list[str], filename: str) -> None:
    for index, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": filename,
                "chunk_index": index
            }]
        )


def retrieve_context(question: str, top_k: int = 6):
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for doc, meta, distance in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": distance
        })

    return retrieved


def build_context(retrieved_chunks: list[dict]) -> str:
    context = ""

    for item in retrieved_chunks:
        context += f"""
Source: {item['source']}
Chunk: {item['chunk_index']}
Content:
{item['text']}

---
"""

    return context