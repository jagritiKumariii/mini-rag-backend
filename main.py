from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mini-rag-index")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set")

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Mini RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Gemini setup (LLM only)
# -------------------------------------------------
genai.configure(api_key=GEMINI_API_KEY)
llm_model = genai.GenerativeModel("gemini-pro")

# -------------------------------------------------
# Local embedding model (CPU ONLY)
# -------------------------------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

EMBEDDING_DIM = 384

# -------------------------------------------------
# Pinecone setup
# -------------------------------------------------
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

# -------------------------------------------------
# Models
# -------------------------------------------------
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

# -------------------------------------------------
# Helpers
# -------------------------------------------------
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


def get_embedding(text: str) -> list[float]:
    emb = embedding_model.encode(
        text,
        normalize_embeddings=True
    )
    return emb.tolist()

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def health():
    return {"status": "healthy", "service": "Mini RAG API"}


@app.post("/api/upload-text")
async def upload_text(data: TextInput):
    start_time = time.time()

    try:
        chunks = chunk_text(data.text)

        vectors = []
        timestamp = int(time.time())

        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            vectors.append({
                "id": f"chunk-{timestamp}-{chunk['id']}",
                "values": embedding,
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
            "time_taken": round(time.time() - start_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()

    try:
        content = await file.read()
        text = content.decode("utf-8")

        chunks = chunk_text(text)

        vectors = []
        timestamp = int(time.time())

        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            vectors.append({
                "id": f"chunk-{timestamp}-{chunk['id']}",
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "chunk_id": chunk["id"],
                    "source": file.filename
                }
            })

        index.upsert(vectors=vectors)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "time_taken": round(time.time() - start_time, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_docs(data: QueryInput):
    start_time = time.time()

    try:
        query_embedding = get_embedding(data.query)

        results = index.query(
            vector=query_embedding,
            top_k=data.top_k,
            include_metadata=True
        )

        if not results.matches:
            return QueryResponse(
                answer="I couldn't find relevant information in the uploaded data.",
                citations=[],
                chunks_retrieved=0,
                time_taken=round(time.time() - start_time, 2)
            )

        context_blocks = []
        citations = []

        for i, match in enumerate(results.matches[:3]):
            text = match.metadata.get("text", "")
            context_blocks.append(f"[{i+1}] {text}")
            citations.append({
                "id": i + 1,
                "source": match.metadata.get("source", "unknown"),
                "score": round(match.score, 3),
                "preview": text[:200] + "..."
            })

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
Use citations like [1], [2].

Context:
{context}

Question:
{data.query}

Answer:
"""

        response = llm_model.generate_content(prompt)
        answer = response.text.strip()

        return QueryResponse(
            answer=answer,
            citations=citations,
            chunks_retrieved=len(results.matches),
            time_taken=round(time.time() - start_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# Local run (ignored by Railway)
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
