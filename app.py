from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.ingenstion import RAGPipelineOrchestrator



# Paths (the repo folder is "Data", so we pass it explicitly to the pipeline)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
VECTOR_DIR = DATA_DIR / "vector_store"
TEMPLATE_FILE = BASE_DIR / "Template" / "index.html"  

# Holds the pipeline once it has loaded (the embedding model is heavy)
state = {"pipeline": None, "error": None}

# Start-up: build the pipeline once, reuse it for every request
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["pipeline"] = RAGPipelineOrchestrator(
            data_directory=str(DATA_DIR),
            persist_directory=str(VECTOR_DIR),
        )
 # keep the server alive so the UI can show the problem
    except Exception as e:

        state["error"] = str(e)
        print(f"Pipeline failed to start: {e}")
    yield


app = FastAPI(title="Research AI Assistant", lifespan=lifespan)

# Lets the UI work even when index.html is opened directly as a file
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> RAGPipelineOrchestrator:
    if state["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail=state["error"] or "The pipeline is still starting. Try again in a moment.",
        )
    return state["pipeline"]

# Request model
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=10)

# Routes
@app.get("/", include_in_schema=False)
def home():
    """Serve the UI."""
    return FileResponse(TEMPLATE_FILE)


@app.get("/api/health")
def health():
    """Server, LLM and database status for the sidebar."""
    pipeline = state["pipeline"]
    if pipeline is None:
        return {"status": "starting", "llm_ready": False, "chunks_in_db": 0, "error": state["error"]}

    return {
        "status": "ok",
        "llm_ready": pipeline.llm is not None,
        "chunks_in_db": pipeline.vector_store.collection.count(),
        "error": None,
    }


@app.get("/api/documents")
def documents():
    """List the PDFs found in the Data folder."""
    files = sorted(DATA_DIR.glob("**/*.pdf"))
    return {
        "documents": [{"name": f.name, "size": f.stat().st_size} for f in files]
    }


@app.post("/api/ask")
def ask(body: AskRequest):
    """
    Same steps as RAGPipelineOrchestrator.run_query():
        1. retrieve the top-K chunks  2. join them into context  3. ask Groq
    We run the steps here (using the pipeline's own retriever and LLM)
    so the response can also include the sources.
    """
    pipeline = get_pipeline()
    question = body.question.strip()

    retrieved = pipeline.retriever.retrieve(question, top_k=body.top_k)

    if not retrieved:
        return {"answer": "No relevant documents found.", "sources": []}

    if not pipeline.llm:
        raise HTTPException(
            status_code=503,
            detail="Groq LLM not initialized. Check your GROQ_API_KEY in the .env file.",
        )

    context = "\n\n---\n\n".join(doc["content"] for doc in retrieved)
    answer = pipeline.llm.generate_response(query=question, context=context)

    sources = [
        {
            "source": doc["metadata"].get("source_file", "unknown"),
            "page": int(doc["metadata"].get("page", 0)) + 1,  # PyPDF pages start at 0
            "score": round(float(doc["similarity_score"]), 4),
            "snippet": doc["content"][:300].strip() + ("…" if len(doc["content"]) > 300 else ""),
        }
        for doc in retrieved
    ]

    return {"answer": answer, "sources": sources}


@app.post("/api/ingest")
def ingest():
    """Read every PDF in Data/, chunk, embed and store it (safe to re-run)."""
    pipeline = get_pipeline()
    try:
        pipeline.ingest_pdfs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total = pipeline.vector_store.collection.count()
    return {"message": f"Done. {total} chunks are now in the database."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
