from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sqlite3
import os
import hashlib
import struct
import numpy as np
import ollama
import faiss
from pydantic import BaseModel
from typing import List, Optional
import json
import time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DB = "rag.db"
CONFIG = "config"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3:latest"
EMBEDDING_DIM = 768  # Fixed dimension for nomic-embed-text

# Models
class QueryRequest(BaseModel):
    question: str

class ConfigUpdate(BaseModel):
    files: List[str]

class StartupResponse(BaseModel):
    needs_update: bool
    added: List[str]
    removed: List[str]
    modified: List[str]

class ContextItem(BaseModel):
    file: str
    start: int
    end: int
    preview: str

class QueryResponse(BaseModel):
    answer: str
    context_used: List[ContextItem]

# Database setup
def init_db():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            hash TEXT,
            last_modified REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            text TEXT,
            start_line INTEGER,
            end_line INTEGER,
            FOREIGN KEY(doc_id) REFERENCES docs(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings_map (
            chunk_id INTEGER,
            faiss_id INTEGER,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id)
        )
    """)
    con.commit()
    con.close()
    print(f"Database initialized: {DB}")

# File operations
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_config():
    if not os.path.exists(CONFIG):
        return []
    with open(CONFIG, "r") as f:
        return [os.path.expanduser(line.strip()) for line in f if line.strip()]

def save_config(files):
    with open(CONFIG, "w") as f:
        for file in files:
            f.write(file + "\n")

# Chunking
def chunk_file(text, file_path):
    chunks = []
    lines = text.splitlines()
    
    if not lines:
        return chunks
    
    # Simple chunking: by fixed number of lines with overlap
    chunk_size = 20
    overlap = 5
    
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunk_text = '\n'.join(lines[i:end])
        
        chunks.append({
            'text': chunk_text,
            'start': i,
            'end': end - 1
        })
        
        i += (chunk_size - overlap)
    
    return chunks

# FAISS operations
def get_faiss_index():
    if os.path.exists("faiss.index"):
        return faiss.read_index("faiss.index")
    else:
        # nomic-embed-text has 768 dimensions
        return faiss.IndexFlatIP(EMBEDDING_DIM)

def save_faiss_index(index):
    faiss.write_index(index, "faiss.index")

# Embedding with dimension validation
def get_embedding(text):
    try:
        print(f"Generating embedding for text ({len(text)} chars)")
        result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        
        if not result or "embedding" not in result:
            raise ValueError("No embedding in response")
        
        embedding = result["embedding"]
        
        # Validate embedding shape
        if len(embedding) != EMBEDDING_DIM:
            print(f"Warning: Embedding has dimension {len(embedding)}, expected {EMBEDDING_DIM}")
            # Pad or truncate to correct dimension
            if len(embedding) > EMBEDDING_DIM:
                embedding = embedding[:EMBEDDING_DIM]
            else:
                embedding = list(embedding) + [0.0] * (EMBEDDING_DIM - len(embedding))
        
        return np.array(embedding, dtype=np.float32)
        
    except Exception as e:
        print(f"Embedding error: {e}")
        # Return a zero vector as fallback with correct dimension
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

# API Endpoints
@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/startup")
async def startup_check():
    """Check if reindexing is needed"""
    try:
        config_files = set(load_config())
        
        con = sqlite3.connect(DB)
        cur = con.cursor()
        cur.execute("SELECT path, hash FROM docs")
        db_files = {row[0]: row[1] for row in cur.fetchall()}
        con.close()
        
        added = config_files - set(db_files.keys())
        removed = set(db_files.keys()) - config_files
        
        modified = []
        for file in config_files & set(db_files.keys()):
            if os.path.exists(file) and file_hash(file) != db_files[file]:
                modified.append(file)
        
        return StartupResponse(
            needs_update=bool(added or removed or modified),
            added=list(added),
            removed=list(removed),
            modified=modified
        )
    except Exception as e:
        print(f"Startup check error: {e}")
        return StartupResponse(needs_update=False, added=[], removed=[], modified=[])

@app.post("/reindex")
async def reindex():
    """Reindex all files"""
    try:
        print("Starting reindexing process...")
        
        # Clean up any existing FAISS index
        if os.path.exists("faiss.index"):
            os.remove("faiss.index")
        
        init_db()
        config_files = load_config()
        
        if not config_files:
            return {"status": "error", "message": "No files in config to index"}
        
        print(f"Found {len(config_files)} files in config")
        
        con = sqlite3.connect(DB)
        cur = con.cursor()
        
        # Clear existing data
        print("Clearing existing data...")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM docs")
        cur.execute("DELETE FROM embeddings_map")
        con.commit()
        
        # Create new FAISS index
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        all_embeddings = []
        chunk_mappings = []
        
        total_chunks = 0
        processed_files = 0
        
        for file_path in config_files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            try:
                print(f"Processing: {file_path}")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                file_hash_val = file_hash(file_path)
                
                # Insert document
                cur.execute(
                    "INSERT OR REPLACE INTO docs (path, hash, last_modified) VALUES (?, ?, ?)",
                    (file_path, file_hash_val, os.path.getmtime(file_path))
                )
                doc_id = cur.lastrowid
                
                # Chunk file
                chunks = chunk_file(text, file_path)
                print(f"Created {len(chunks)} chunks")
                
                for chunk in chunks:
                    cur.execute(
                        "INSERT INTO chunks (doc_id, text, start_line, end_line) VALUES (?, ?, ?, ?)",
                        (doc_id, chunk['text'], chunk['start'], chunk['end'])
                    )
                    chunk_id = cur.lastrowid
                    
                    # Generate embedding
                    embedding = get_embedding(chunk['text'])
                    
                    # Validate embedding dimension before adding
                    if embedding.shape[0] != EMBEDDING_DIM:
                        print(f"Error: Embedding dimension mismatch: {embedding.shape[0]} != {EMBEDDING_DIM}")
                        continue
                    
                    all_embeddings.append(embedding)
                    chunk_mappings.append(chunk_id)
                    
                    total_chunks += 1
                    
                    if total_chunks % 100 == 0:
                        print(f"Processed {total_chunks} chunks...")
                
                processed_files += 1
                print(f"Completed {processed_files}/{len(config_files)} files")
        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Processed {processed_files}/{len(config_files)} files")
        print(f"Total chunks created: {total_chunks}")
        print(f"Total embeddings generated: {len(all_embeddings)}")
        
        # Add all embeddings to FAISS
        if all_embeddings:
            try:
                # Convert to numpy array with proper shape checking
                print("Converting embeddings to numpy array...")
                embeddings_array = np.array(all_embeddings, dtype=np.float32)
                print(f"Embeddings array shape: {embeddings_array.shape}")
                print(f"Embeddings array dtype: {embeddings_array.dtype}")
                
                if embeddings_array.shape[1] != EMBEDDING_DIM:
                    print(f"Error: Array shape mismatch. Expected {EMBEDDING_DIM} columns, got {embeddings_array.shape[1]}")
                    return {"status": "error", "message": f"Embedding dimension mismatch: {embeddings_array.shape[1]} != {EMBEDDING_DIM}"}
                
                print(f"Adding {len(embeddings_array)} embeddings to FAISS index...")
                index.add(embeddings_array)
                save_faiss_index(index)
                print(f"FAISS index now contains {index.ntotal} vectors")
                
                # Store chunk to FAISS ID mapping
                print("Creating chunk to FAISS ID mappings...")
                for faiss_id, chunk_id in enumerate(chunk_mappings):
                    cur.execute(
                        "INSERT INTO embeddings_map (chunk_id, faiss_id) VALUES (?, ?)",
                        (chunk_id, faiss_id)
                    )
                print(f"Created {len(chunk_mappings)} mappings")
                
            except Exception as e:
                print(f"Error creating FAISS index: {e}")
                import traceback
                traceback.print_exc()
                return {"status": "error", "message": f"FAISS index creation failed: {str(e)}"}
        else:
            print("No embeddings to add to FAISS index")
        
        con.commit()
        con.close()
        
        # Verify FAISS file was created
        if os.path.exists("faiss.index"):
            file_size = os.path.getsize("faiss.index")
            print(f"FAISS index file created: {file_size} bytes")
            return {
                "status": "success", 
                "message": f"Indexed {processed_files} files, created {total_chunks} chunks, FAISS vectors: {index.ntotal}"
            }
        else:
            print("FAISS index file was NOT created!")
            return {"status": "error", "message": "FAISS index file was not created"}
    
    except Exception as e:
        print(f"Reindex error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/query")
async def query(request: QueryRequest):
    """Handle RAG query"""
    try:
        if not os.path.exists("faiss.index"):
            print("FAISS index not found - reindex required")
            raise HTTPException(status_code=400, detail="Index not built. Please reindex first.")
        
        print(f"Query: {request.question}")
        
        # Check FAISS index size
        index = get_faiss_index()
        if index.ntotal == 0:
            print("FAISS index is empty - reindex required")
            raise HTTPException(status_code=400, detail="Index is empty. Please reindex first.")
        
        print(f"FAISS index has {index.ntotal} vectors")
        
        # Embed query
        query_embedding = get_embedding(request.question)
        if query_embedding.shape[0] != EMBEDDING_DIM:
            print(f"Query embedding dimension mismatch: {query_embedding.shape[0]} != {EMBEDDING_DIM}")
            raise HTTPException(status_code=500, detail="Query embedding dimension mismatch")
        
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS
        k = min(10, index.ntotal)
        print(f"Searching for {k} nearest neighbors...")
        D, I = index.search(query_embedding, k=k)
        
        print(f"Search results - Found {len(I[0])} matches")
        
        # Get chunk details
        con = sqlite3.connect(DB)
        cur = con.cursor()
        
        context_chunks = []
        context_used = []
        
        for i, faiss_id in enumerate(I[0]):
            if faiss_id < 0:  # Invalid ID
                continue
                
            # Get chunk_id from mapping
            cur.execute("SELECT chunk_id FROM embeddings_map WHERE faiss_id = ?", (int(faiss_id),))
            mapping_result = cur.fetchone()
            
            if not mapping_result:
                continue
                
            chunk_id = mapping_result[0]
            
            cur.execute("""
                SELECT c.text, c.start_line, c.end_line, d.path 
                FROM chunks c 
                JOIN docs d ON c.doc_id = d.id 
                WHERE c.id = ?
            """, (chunk_id,))
            
            result = cur.fetchone()
            if result:
                text, start, end, path = result
                print(f"Found context: {path} lines {start}-{end}")
                # Format context with file info
                formatted_context = f"[file: {path} | lines {start}-{end}]\n{text}"
                context_chunks.append(formatted_context)
                context_used.append(ContextItem(
                    file=path,
                    start=start,
                    end=end,
                    preview=text[:200] + "..." if len(text) > 200 else text
                ))
        
        if not context_chunks:
            print("No context chunks found for query")
            raise HTTPException(status_code=400, detail="No context found for query")
        
        print(f"Found {len(context_chunks)} context chunks")
        
        # Build prompt
        context_text = "\n\n".join(context_chunks)
        prompt = f"""Context information:
{context_text}

Based on the above context, please answer this question:

Question: {request.question}

Answer:"""
        
        print(f"Prompt length: {len(prompt)} characters")
        
        # Get LLM response
        print("Generating LLM response...")
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        answer = response["message"]["content"]
        print(f"LLM response generated: {len(answer)} characters")
        
        # Save to history
        cur.execute(
            "INSERT INTO history (question, answer) VALUES (?, ?)",
            (request.question, answer)
        )
        con.commit()
        con.close()
        
        return QueryResponse(answer=answer, context_used=context_used)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/config")
async def get_config():
    return {"files": load_config()}

@app.post("/config")
async def update_config(config: ConfigUpdate):
    save_config(config.files)
    return {"status": "success"}

@app.delete("/history")
async def clear_history():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("DELETE FROM history")
    con.commit()
    con.close()
    return {"status": "success"}

@app.get("/history")
async def get_history():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT question, answer, timestamp FROM history ORDER BY timestamp DESC LIMIT 50")
    history = [{"question": row[0], "answer": row[1], "timestamp": row[2]} for row in cur.fetchall()]
    con.close()
    return history

# Serve static files
app.mount("/", StaticFiles(directory="frontend"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Initialize database
    init_db()
    print("Starting RAG server on http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
