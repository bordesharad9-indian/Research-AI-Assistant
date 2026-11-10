
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    """Generates embeddings using SentenceTransformer."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        self.model_name = model_name

        print(f"Loading embedding model: {self.model_name}")

        self.model = SentenceTransformer(self.model_name)

        print(
            f"Model loaded successfully. "
            f"Embedding dimension: {self.model.get_embedding_dimension()}"
        )

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:

        if not texts:
            return np.empty(
                (0, self.model.get_embedding_dimension()),
                dtype=np.float32
            )

        print(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(texts,show_progress_bar=True,normalize_embeddings=True)

        print(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings

class VectorStore:
    """Stores document embeddings using ChromaDB."""

    def __init__(self,collection_name="pdf_documents",persist_directory=None):

        self.collection_name = collection_name

        # Set absolute project directory
        project_dir = Path(__file__).resolve().parent.parent

        if persist_directory is None:
            persist_directory = (
                project_dir / "data" / "vector_store"
            )

        self.persist_directory = str(
            Path(persist_directory).resolve()
        )

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PDF document embeddings for RAG",
                "hnsw:space": "cosine"
            }
        )

        print(
            f"Vector store initialized. "
            f"Collection: {self.collection_name}"
        )

        print(
            f"Existing documents in collection: "
            f"{self.collection.count()}"
        )

    def add_documents(self,documents: List[Any],embeddings: np.ndarray):

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        if not documents:
            print("No documents to add.")
            return

        print(f"Adding {len(documents)} document chunks...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for doc, embedding in zip(documents, embeddings):

            content = doc.page_content

            source = str(
                doc.metadata.get("source_file", "unknown")
            )

            page = doc.metadata.get("page", 0)

            # Stable document ID
            unique_text = f"{source}_{page}_{content}"

            doc_id = hashlib.sha256(
                unique_text.encode("utf-8")
            ).hexdigest()

            metadata = {}

            for key, value in doc.metadata.items():

                if value is not None and isinstance(
                    value, (str, int, float, bool)
                ):
                    metadata[key] = value

            metadata["content_length"] = len(content)

            ids.append(doc_id)
            metadatas.append(metadata)
            documents_text.append(content)

            embeddings_list.append(
                np.asarray(embedding).tolist()
            )

        # Store documents in batches
        batch_size = 500

        for start in range(0, len(ids), batch_size):

            end = start + batch_size

            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings_list[start:end],
                metadatas=metadatas[start:end],
                documents=documents_text[start:end]
            )

        print("Documents successfully stored!")

        print(
            f"Total documents in collection: "
            f"{self.collection.count()}"
        )

class RAGRetriever:
    """Retrieves relevant documents from ChromaDB."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager
    ):

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self,query: str,top_k: int = 5,score_threshold: float = 0.0) -> List[Dict[str, Any]]:

        print(f"Retrieving documents for query: '{query}'")

        collection = self.vector_store.collection

        total_documents = collection.count()

        if total_documents == 0:
            print("Vector database is empty.")
            return []

        # Generate query embedding
        query_embedding = (
            self.embedding_manager.generate_embeddings([query])[0]
        )

        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, total_documents),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_docs = []

        if not results["documents"] or not results["documents"][0]:
            print("No relevant documents found.")
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        for rank, (
            doc_id,
            document,
            metadata,
            distance
        ) in enumerate(
            zip(ids, documents, metadatas, distances),
            start=1
        ):

            # Cosine similarity score
            similarity_score = 1.0 - distance

            if similarity_score >= score_threshold:

                retrieved_docs.append({

                    "id": doc_id,

                    "content": document,

                    "metadata": metadata,

                    "similarity_score": similarity_score,

                    "distance": distance,

                    "rank": rank

                })

        print(
            f"Retrieved {len(retrieved_docs)} relevant documents."
        )

        return retrieved_docs