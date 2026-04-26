import os
import shutil
import uuid
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import DATA_DIR
from rag_engine import engine, RelevantImage

class ImagePayload(BaseModel):
    source_file: str
    page_num: int
    image_kind: str
    relevance_score: float
    mime_type: str
    data: str  # base64


class ChatResponse(BaseModel):
    query: str
    response: str
    images: List[ImagePayload]

app = FastAPI(title="Multimodal RAG Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    # A simple one liner health check
    return {"status": "ok", "message": "Multimodal RAG API is live."}

@app.post("/ingest")
async def upload_and_ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Create a unique subfolder for this ingestion batch to prevent overwriting
    batch_id = str(uuid.uuid4())
    batch_dir = DATA_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Verbose step by step file saving loop
    for uploaded_file in files:
        file_path = batch_dir / uploaded_file.filename
        
        try:
            with open(file_path, "wb") as buffer:
                # Copying the file block by block, standard practice for fastAPI
                shutil.copyfileobj(uploaded_file.file, buffer)
            
            saved_files.append(uploaded_file.filename)
        except Exception as e:
            # We fail out completely rather than partially ingesting
            raise HTTPException(status_code=500, detail=f"Failed to save {uploaded_file.filename}: {e}")
            
    # Now ask the engine to ingest the directory we just made
    success = engine.ingest_documents(str(batch_dir))
    
    if not success:
        return JSONResponse(status_code=500, content={"error": "Failed to extract/index documents."})
    
    return {
        "message": "Ingestion successful",
        "batch_id": batch_id,
        "files_indexed": saved_files
    }

from query_router import query_router

@app.post("/chat", response_model=ChatResponse)
def chat_with_bot(query: str, temperature: float = 0.7, max_new_tokens: int = 500):
    if not query or query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = query_router.route_query(query, temperature=temperature, max_new_tokens=max_new_tokens)

        images = [
            ImagePayload(
                source_file=img.source_file,
                page_num=img.page_num,
                image_kind=img.image_kind,
                relevance_score=img.relevance_score,
                mime_type=img.mime_type,
                data=img.image_b64,
            )
            for img in result.images
        ]

        return ChatResponse(query=query, response=result.answer, images=images)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
