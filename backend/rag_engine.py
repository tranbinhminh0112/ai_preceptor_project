"""
Retrieval-Augmented Generation engine for the nursing assistant.

Wraps the exact pipeline from ai_model_interaction_v2.ipynb:
  - bge-base embeddings + persistent Chroma vector DB
  - Hybrid retriever (BM25 keyword + vector semantic search)
  - Qwen 2.5 (via Ollama) with a strict, no-hallucination nursing prompt

The heavy objects (embeddings, vector DB, retriever) are built once and
reused. In Streamlit they are wrapped with @st.cache_resource by the caller.
"""

from __future__ import annotations

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama

from config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE_ANSWER,
    RETRIEVER_TOP_K,
    HYBRID_WEIGHTS,
)

SYSTEM_PROMPT_TEMPLATE = """
You are a Senior Nursing Instructor at the Institute of Mental Health (IMH). Your primary task is to answer questions based strictly on the provided medical context.

RULES:
1. NO HALLUCINATION: Use ONLY the information provided in the [CONTEXT] below. If the answer cannot be deduced from the context, state clearly that you do not have the information. Do not invent facts.
2. FORMATTING: If the information involves comparisons, levels, or clinical categorizations, present it using a clean Markdown Table.
3. EMPHASIS: Always bold key psychiatric and medical terms such as **MFAAS**, **MSE**, **Hallucination**, etc.
4. TONE: Provide a professional, concise, and clinically accurate response suitable for clinical nursing staff.
5. LANGUAGE CONSTRAINT: You must ALWAYS respond in the EXACT SAME LANGUAGE as the user's query. UNDER NO CIRCUMSTANCES should you output Chinese unless the user explicitly asks a question in Chinese.

[CONTEXT]:
{context}
"""


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embeddings():
    """
    Load the embedding model (must match the one used to build the DB).

    Prefer the locally cached copy (fast, works offline). If it has never been
    downloaded, fall back to fetching it from Hugging Face on first run so the
    app works out-of-the-box after a fresh clone.
    """
    device = detect_device()
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device, "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        # Not cached yet — download it (needs internet, one time only).
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )


def load_vector_db(embeddings):
    """Connect to the persistent Chroma vector database."""
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def build_hybrid_retriever(vector_db):
    """Create the BM25 + vector ensemble (hybrid) retriever."""
    all_data = vector_db.get()
    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(all_data["documents"], all_data["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = RETRIEVER_TOP_K

    vector_retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=HYBRID_WEIGHTS,
    )


class RAGEngine:
    """Bundles the retriever + LLM and exposes the nursing Q&A behaviour."""

    def __init__(self):
        self.embeddings = load_embeddings()
        self.vector_db = load_vector_db(self.embeddings)
        self.retriever = build_hybrid_retriever(self.vector_db)

    @property
    def vector_count(self) -> int:
        try:
            return self.vector_db._collection.count()
        except Exception:
            return 0

    def retrieve(self, query: str):
        """Return the hybrid-retrieved documents for a query."""
        return self.retriever.invoke(query)

    def answer(self, query: str):
        """
        Answer a clinical question.

        Returns (answer_text, retrieved_docs). retrieved_docs may be empty.
        """
        retrieved_docs = self.retrieve(query)
        if not retrieved_docs:
            return "No relevant information found in the database.", []

        context_list = []
        for i, doc in enumerate(retrieved_docs):
            hierarchy = doc.metadata.get("hierarchy", "N/A")
            content = doc.metadata.get("original_content", doc.page_content)
            context_list.append(f"[Source {i + 1}: {hierarchy}]\n{content}")
        full_context = "\n\n---\n\n".join(context_list)

        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE_ANSWER)
        response = llm.invoke(
            [
                ("system", SYSTEM_PROMPT_TEMPLATE.format(context=full_context)),
                ("human", query),
            ]
        )
        return response.content, retrieved_docs


def group_sources(retrieved_docs):
    """
    Group retrieved docs by source document for clean reference rendering.

    Returns: list of {"source": str, "sections": [{"hierarchy", "asset"}, ...]}
    """
    unique = {}
    for doc in retrieved_docs or []:
        source = doc.metadata.get("source", "Unknown Document")
        hierarchy = doc.metadata.get("hierarchy", "Unknown Section")
        asset = doc.metadata.get("asset", "None")
        unique.setdefault(source, set()).add((hierarchy, asset))

    grouped = []
    for source, details in unique.items():
        sections = [
            {"hierarchy": h, "asset": (a if a and str(a).lower() != "none" else None)}
            for h, a in sorted(details, key=lambda x: x[0])
        ]
        grouped.append({"source": source, "sections": sections})
    return grouped
