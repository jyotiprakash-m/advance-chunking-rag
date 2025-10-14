from fastapi import FastAPI, UploadFile, File
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from api.v1.structural_block import router as structural_block_router

# Import structural block chunking service
# Note: Due to hyphens in directory name, we'll import at runtime

app = FastAPI(title="RAG Simulator", description="A FastAPI application for RAG simulation", version="1.0.0")

# Cors policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# functionality of chunking
app.include_router(structural_block_router, prefix="/v1")

@app.get("/")
def read_root():
    return {"message": "Hello from rag-simulator!"}

@app.get("/health")
def health_check():
    """Health check endpoint for Docker container monitoring."""
    return {"status": "healthy", "service": "rag-chunking-api"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
