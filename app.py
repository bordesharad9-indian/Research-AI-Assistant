
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# from src.ingenstion import RAGPipelineOrchestrator

# INITIALIZE FASTAPI
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vector_store"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Research Assistant",
    description=(
        "PDF-based research assistant using "
        "Machine Learning, NLP, Deep Learning and RAG"
    ),
    version="1.0.0"
)
templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)
# INITIALIZE RAG PIPELINE
rag = RAGPipelineOrchestrator(
    data_directory=str(UPLOAD_DIR),
    persist_directory=str(VECTOR_DIR)
)
# REQUEST MODEL
class ResearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    companies: list[str] = Field(default_factory=list)
# HOME PAGE
@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )
# PDF UPLOAD
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    filename = Path(file.filename).name
    file_path = UPLOAD_DIR / filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "message": "PDF uploaded successfully.",
            "filename": filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        await file.close()
# PDF INGESTION
@app.post("/ingest")
def ingest_documents():

    try:

        rag.ingest_pdfs(
            chunk_size=10000,
            chunk_overlap=400
        )

        return {
            "status": "success",
            "message": "PDF ingestion completed successfully."
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# RESEARCH QUESTION ANSWERING
# ==========================================

@app.post("/research")
def execute_research(req: ResearchRequest):

    if not req.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Please enter a research question."
        )

    try:

        if not rag.llm:

            raise HTTPException(
                status_code=503,
                detail="Groq LLM not initialized. Check GROQ_API_KEY."
            )

        # Retrieve relevant PDF chunks
        docs = rag.retriever.retrieve(
            req.query,
            top_k=req.top_k
        )

        # Filter by company if metadata is available
        wanted = {
            company.lower()
            for company in req.companies
        }

        if wanted:

            docs = [
                d for d in docs
                if (
                    not d.get("metadata", {}).get("company")
                    or d["metadata"]["company"].lower() in wanted
                )
            ]

        if not docs:

            return {
                "query": req.query,
                "report": "No relevant research documents found.",
                "sources": []
            }

        # Prepare retrieved context
        context = "\n\n---\n\n".join(
            d["content"] for d in docs
        )

        # Generate research-oriented response
        report = rag.llm.generate_response(
            query=req.query,
            context=context
        )

        # Prepare source information
        sources = []

        for d in docs:

            metadata = d.get("metadata", {})

            sources.append({
                "source_file": metadata.get(
                    "source_file", "unknown"
                ),
                "page": metadata.get("page"),
                "company": metadata.get("company"),
                "year": metadata.get("year"),
                "similarity_score": round(
                    float(d.get("similarity_score", 0)), 3
                )
            })

        return {
            "status": "success",
            "query": req.query,
            "report": report,
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "running",
        "message": "AI Research Assistant is working."
    }
# RUN APPLICATION
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
