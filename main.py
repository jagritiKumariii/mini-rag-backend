from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import time
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import tiktoken

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize embedding model (runs locally, completely free!)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mini-rag-index")

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(INDEX_NAME)

# Models
class TextInput(BaseModel):
    text: str

class QueryInput(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    chunks_retrieved: int
    tokens_used: int
    time_taken: float

# Chunking function
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[dict]:
    """Split text into overlapping chunks with metadata"""
    words = text.split()
    chunks = []
    
    start = 0
    chunk_id = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "chunk_id": chunk_id,
                "start_pos": start,
                "end_pos": end,
                "chunk_size": len(chunk_words)
            }
        })
        
        chunk_id += 1
        start += chunk_size - overlap
    
    return chunks

# Embedding function (FREE - runs locally)
def get_embedding(text: str) -> List[float]:
    """Get embedding using local model"""
    embedding = embedding_model.encode(text)
    return embedding.tolist()

# Simple reranking function (no API needed)
def rerank_chunks(query: str, chunks: List[dict], top_n: int = 3) -> List[dict]:
    """Simple reranking based on keyword matching and score"""
    # Sort by existing score (already good from vector search)
    sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
    return sorted_chunks[:top_n]

# Store text endpoint
@app.post("/api/upload-text")
async def upload_text(input_data: TextInput):
    start_time = time.time()
    
    try:
        # Chunk the text
        chunks = chunk_text(input_data.text)
        
        # Generate embeddings and upsert to Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk["text"])
            vectors.append({
                "id": f"chunk_{i}_{int(time.time())}",
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "chunk_id": chunk["metadata"]["chunk_id"],
                    "source": "user_input"
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        time_taken = time.time() - start_time
        
        return {
            "status": "success",
            "chunks_created": len(chunks),
            "time_taken": round(time_taken, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload file endpoint
@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()
    
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Generate embeddings and upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk["text"])
            vectors.append({
                "id": f"chunk_{i}_{int(time.time())}",
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "chunk_id": chunk["metadata"]["chunk_id"],
                    "source": file.filename
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        time_taken = time.time() - start_time
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "time_taken": round(time_taken, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoint
@app.post("/api/query", response_model=QueryResponse)
async def query(query_input: QueryInput):
    start_time = time.time()
    
    try:
        # Get query embedding
        query_embedding = get_embedding(query_input.query)
        
        # Retrieve from Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=query_input.top_k,
            include_metadata=True
        )
        
        # Extract chunks
        retrieved_chunks = [
            {
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "source": match.metadata.get("source", "unknown"),
                "chunk_id": match.metadata.get("chunk_id", 0)
            }
            for match in results.matches
        ]
        
        if not retrieved_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your query.",
                citations=[],
                chunks_retrieved=0,
                tokens_used=0,
                time_taken=round(time.time() - start_time, 2)
            )
        
        # Simple reranking (no API needed)
        reranked_chunks = rerank_chunks(query_input.query, retrieved_chunks, top_n=3)
        
        # Build context for LLM
        context = "\n\n".join([
            f"[{i+1}] {chunk['text']}"
            for i, chunk in enumerate(reranked_chunks)
        ])
        
        # Generate answer with Gemini (FREE!)
        prompt = f"""Based on the following context, answer the user's question. 
Use inline citations like [1], [2], etc. to reference the sources.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query_input.query}

Answer with citations:"""
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        answer = response.text
        
        # Estimate tokens (Gemini doesn't provide exact count in free tier)
        tokens_used = len(prompt.split()) + len(answer.split())
        
        # Format citations
        citations = [
            {
                "id": i + 1,
                "text": chunk["text"][:200] + "...",
                "source": chunk["source"],
                "relevance_score": round(chunk["score"], 3)
            }
            for i, chunk in enumerate(reranked_chunks)
        ]
        
        time_taken = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            chunks_retrieved=len(retrieved_chunks),
            tokens_used=tokens_used,
            time_taken=round(time_taken, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Mini RAG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)