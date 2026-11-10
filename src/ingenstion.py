
import os
from pathlib import Path
from typing import List, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.agent_workflow import GroqLLM

from src.store_croma_db import (
    EmbeddingManager,
    VectorStore,
    RAGRetriever
)


class RAGPipelineOrchestrator:
    """End-to-end RAG pipeline."""

    def __init__(
        self,
        data_directory=None,
        persist_directory=None
    ):

        base_dir = Path(__file__).resolve().parent.parent

        self.data_directory = data_directory or str(
            base_dir / "data"
        )

        persist_directory = persist_directory or str(
            base_dir / "data" / "vector_store"
        )

        self.embedding_manager = EmbeddingManager()

        self.vector_store = VectorStore(
            persist_directory=persist_directory
        )

        self.retriever = RAGRetriever(
            self.vector_store,
            self.embedding_manager
        )

        try:
            self.llm = GroqLLM()
        except ValueError as e:
            print(f"Warning: {e}")
            self.llm = None

    def ingest_pdfs(
        self,
        chunk_size=10000,
        chunk_overlap=400
    ):

        pdf_dir = Path(self.data_directory)

        if not pdf_dir.exists():
            raise FileNotFoundError(
                f"PDF directory not found: {pdf_dir}"
            )

        pdf_files = list(pdf_dir.glob("**/*.pdf"))

        print(f"Found {len(pdf_files)} PDF files")

        if not pdf_files:
            print("No PDF files found!")
            return

        all_documents = []

        for pdf_file in pdf_files:

            print(f"\nProcessing: {pdf_file.name}")

            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["file_type"] = "pdf"

                all_documents.extend(documents)

                print(f"Loaded {len(documents)} pages")

            except Exception as e:
                print(f"Error loading {pdf_file.name}: {e}")

        if not all_documents:
            print("No pages extracted from PDFs.")
            return

        print(f"\nTotal pages loaded: {len(all_documents)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = text_splitter.split_documents(
            all_documents
        )

        print(f"Total chunks created: {len(split_docs)}")

        texts = [
            doc.page_content for doc in split_docs
        ]

        embeddings = self.embedding_manager.generate_embeddings(
            texts
        )

        self.vector_store.add_documents(
            split_docs,
            embeddings
        )

        print("PDF ingestion completed successfully!")

    def run_query(self, query: str, top_k: int = 5):

        retrieved_docs = self.retriever.retrieve(
            query,
            top_k=top_k
        )

        if not retrieved_docs:
            return "No relevant documents found."

        context = "\n\n---\n\n".join(
            [doc["content"] for doc in retrieved_docs]
        )

        if not self.llm:
            return "Groq LLM not initialized. Check your API key."

        response = self.llm.generate_response(
            query=query,
            context=context
        )

        return response


if __name__ == "__main__":

    pipeline = RAGPipelineOrchestrator()

    # STEP 1: Load and store PDFs
    pipeline.ingest_pdfs()

    # STEP 2: Ask a question
    query = "Analyze the uploaded PDF documents using NLP for text extraction and semantic understanding, Random Forest for feature selection and classification, Deep Learning for identifying complex patterns, and Retrieval-Augmented Generation (RAG) for retrieving relevant information and generating accurate, context-aware answers based on the document content."

    answer = pipeline.run_query(query)
    print(answer)