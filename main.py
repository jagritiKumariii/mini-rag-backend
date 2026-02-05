from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# =================================================
# Load environment variables
# =================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mini-rag-index")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

# =================================================
# FastAPI app
# =================================================
app = FastAPI(title="Mini RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================================
# Local embedding model (CPU ONLY)
# =================================================
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

EMBEDDING_DIM = 384

def get_embedding(text: str) -> List[float]:
    emb = embedding_model.encode(text, normalize_embeddings=True)
    return emb.tolist()

# =================================================
# Pinecone setup
# =================================================
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(PINECONE_INDEX_NAME)

# =================================================
# Models
# =================================================
class TextInput(BaseModel):
    text: str

class QueryInput(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    chunks_retrieved: int
    time_taken: float

# =================================================
# Helpers
# =================================================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    cid = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append({
            "id": cid,
            "text": " ".join(chunk_words)
        })

        cid += 1
        start += chunk_size - overlap

    return chunks

# =================================================
# Routes
# =================================================
@app.get("/")
def health():
    return {"status": "healthy", "service": "Mini RAG API"}

@app.post("/api/upload-text")
async def upload_text(data: TextInput):
    start = time.time()
    chunks = chunk_text(data.text)
    ts = int(time.time())

    vectors = []
    for chunk in chunks:
        vectors.append({
            "id": f"chunk-{ts}-{chunk['id']}",
            "values": get_embedding(chunk["text"]),
            "metadata": {
                "text": chunk["text"],
                "chunk_id": chunk["id"],
                "source": "user_text"
            }
        })

    index.upsert(vectors=vectors)

    return {
        "status": "success",
        "chunks_created": len(chunks),
        "time_taken": round(time.time() - start, 2)
    }

@app.post("/api/query", response_model=QueryResponse)
async def query_docs(data: QueryInput):
    start = time.time()

    query_emb = get_embedding(data.query)
    results = index.query(
        vector=query_emb,
        top_k=data.top_k,
        include_metadata=True
    )

    if not results.matches:
        return QueryResponse(
            answer="No relevant information found.",
            citations=[],
            chunks_retrieved=0,
            time_taken=round(time.time() - start, 2)
        )

    citations = []
    context = []

    for i, match in enumerate(results.matches[:3]):
        text = match.metadata["text"]
        context.append(f"[{i+1}] {text}")
        citations.append({
            "id": i + 1,
            "score": round(match.score, 3),
            "preview": text[:200] + "..."
        })

    answer = "\n\n".join(context)

    return QueryResponse(
        answer=answer,
        citations=citations,
        chunks_retrieved=len(results.matches),
        time_taken=round(time.time() - start, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
