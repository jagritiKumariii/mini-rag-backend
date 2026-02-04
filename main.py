from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
import time
from pinecone import Pinecone, ServerlessSpec
import openai
import tiktoken
import cohere

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
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
co = cohere.Client(os.getenv("COHERE_API_KEY"))

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mini-rag-index")

# Initialize Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI embedding dimension
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
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[dict]:
    """Split text into overlapping chunks with metadata"""
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    chunk_id = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "chunk_id": chunk_id,
                "start_char": start,
                "end_char": end,
                "chunk_size": len(chunk_tokens)
            }
        })
        
        chunk_id += 1
        start += chunk_size - overlap
    
    return chunks

# Embedding function
def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding for text"""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

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
        
        # Rerank with Cohere
        rerank_docs = [chunk["text"] for chunk in retrieved_chunks]
        rerank_response = co.rerank(
            model="rerank-english-v3.0",
            query=query_input.query,
            documents=rerank_docs,
            top_n=3
        )
        
        # Get top reranked chunks
        reranked_chunks = [retrieved_chunks[result.index] for result in rerank_response.results]
        
        # Build context for LLM
        context = "\n\n".join([
            f"[{i+1}] {chunk['text']}"
            for i, chunk in enumerate(reranked_chunks)
        ])
        
        # Generate answer with GPT-4
        prompt = f"""Based on the following context, answer the user's question. 
Use inline citations like [1], [2], etc. to reference the sources.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query_input.query}

Answer with citations:"""
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources using [1], [2], etc."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
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