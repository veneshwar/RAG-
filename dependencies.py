"""
Dependency injection for FastAPI routes
"""

import os
from typing import AsyncGenerator
from functools import lru_cache

from app.rag.orchestrator import RAGOrchestrator
from app.rag.retriever import Retriever
from app.rag.reranker import Reranker
from app.rag.generator import Generator
from app.rag.prompt_builder import PromptBuilder
from app.rag.context_manager import ContextManager
from app.indexing.embeddings.openai_embedder import OpenAIEmbedder
from app.indexing.vectorstores.faiss_store import FAISSVectorStore
from app.models.openai_client import OpenAIClient
from app.data.memory_store import MemoryDocumentStore


@lru_cache()
def get_settings():
    """Get cached settings"""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", "sk-mock-key"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "llm_model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./data/vectorstore"),
    }


async def get_embedder():
    """Get embedder instance"""
    settings = get_settings()
    return OpenAIEmbedder(
        api_key=settings["openai_api_key"],
        model=settings["embedding_model"]
    )


async def get_vector_store():
    """Get vector store instance"""
    settings = get_settings()
    return FAISSVectorStore(
        store_path=settings["vector_store_path"],
        dimension=1536  # OpenAI embedding dimension
    )


async def get_llm_client():
    """Get LLM client instance"""
    settings = get_settings()
    return OpenAIClient(
        api_key=settings["openai_api_key"],
        model=settings["llm_model"]
    )


async def get_document_store():
    """Get document store instance"""
    return MemoryDocumentStore()


async def get_retriever() -> Retriever:
    """Get retriever instance"""
    embedder = await get_embedder()
    vector_store = await get_vector_store()
    return Retriever(
        embedder=embedder,
        vector_store=vector_store,
        similarity_threshold=0.7,
        max_results=100
    )


async def get_reranker() -> Reranker:
    """Get reranker instance"""
    return Reranker(llm_client=None)


async def get_generator() -> Generator:
    """Get generator instance"""
    llm_client = await get_llm_client()
    return Generator(
        llm_client=llm_client,
        max_tokens=1000,
        temperature=0.7
    )


async def get_prompt_builder() -> PromptBuilder:
    """Get prompt builder instance"""
    return PromptBuilder()


async def get_context_manager() -> ContextManager:
    """Get context manager instance"""
    document_store = await get_document_store()
    return ContextManager(document_store=document_store)


async def get_rag_orchestrator() -> RAGOrchestrator:
    """Get fully configured RAG orchestrator"""
    retriever = await get_retriever()
    reranker = await get_reranker()
    generator = await get_generator()
    prompt_builder = await get_prompt_builder()
    context_manager = await get_context_manager()
    
    return RAGOrchestrator(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        prompt_builder=prompt_builder,
        context_manager=context_manager
    )
