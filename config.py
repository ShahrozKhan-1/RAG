import os
from pathlib import Path
import chromadb
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv() 



CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
CHROMA_COLLECTION = "financial_pdfs"

MODEL_URL = "http://localhost:12434/engines/llama.cpp/v1/embeddings"
MODEL_NAME = "ai/nomic-embed-text-v1.5"

gemini = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), google_api_key=os.getenv("GEMINI_API_KEY"))

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
vector_Store = client.get_or_create_collection(name=CHROMA_COLLECTION)



def llm(query: str, system_prompt: str | None = None):
    payload = {
        "model": "ai/qwen3:1.7b-q4_K_M",
        "prompt": query,
        "stream": False,
        "think": False,
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    response = requests.post(
        "http://localhost:12434/api/generate",
        json=payload,
    )
    return response.json()





def embedding(text: str) -> list[float]:
    try:
        response = requests.post(
            MODEL_URL,
            json={
                "model": MODEL_NAME,
                "input": text,
            },
            timeout=120,
        )

        if not response.ok:
            print("\nEmbedding request failed!")
            print("Input length:", len(text), "characters")
            print("Status code:", response.status_code)
            print("Server error:", response.text[:2000])
            print("Input preview:", repr(text[:300]))

            response.raise_for_status()

        result = response.json()
        data = result.get("data", [])

        if len(data) != 1:
            raise ValueError(
                f"Expected 1 embedding, received {len(data)}"
            )

        vector = data[0].get("embedding")

        if not isinstance(vector, list) or not vector:
            raise ValueError("API returned an invalid embedding vector")

        return vector

    except requests.exceptions.Timeout:
        print("Embedding request timed out.")
        raise

    except requests.exceptions.ConnectionError:
        print("Could not connect to the embedding server.")
        raise



def gemini_llm(query: str, system_prompt: str | None = None):
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    response = gemini.invoke(input=messages)
    return response.content[0]['text']