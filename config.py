import os
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
CHROMA_COLLECTION = "financial_pdfs"

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

gemini_chat_model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
vector_store = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)


def embedding(text: str) -> list[float]:
    response = requests.post(
        EMBEDDING_API_URL,
        json={
            "model": EMBEDDING_MODEL_NAME,
            "input": text,
        },
        timeout=120,
    )
    response.raise_for_status()

    response_data = response.json()
    embeddings = response_data.get("data", [])

    if len(embeddings) != 1:
        raise ValueError(f"Expected 1 embedding, received {len(embeddings)}")

    embedding_vector = embeddings[0].get("embedding")
    if not isinstance(embedding_vector, list) or not embedding_vector:
        raise ValueError("API returned an invalid embedding vector")

    return embedding_vector


def gemini_llm(query: str, system_prompt: str | None = None):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    response = gemini_chat_model.invoke(input=messages)
    return response.content[0]["text"]