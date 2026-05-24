"""Orchestrates retrieve -> generate -> validate.

Phase 3 & 4:
- Chain retriever -> generator -> FW-L2 validation
- Takes a user query, retrieves context, generates answer
- Optionally validates and redacts PII from the response (FW-L2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config import TOP_K
from app.rag.retriever import Retriever, RetrievedChunk
from app.rag.generator import Generator, GeneratorResponse
from app.firewall.fw_l2 import FWL2, FWL2Result


@dataclass
class RAGResponse:
    """Full RAG pipeline response."""

    query: str
    answer: str
    raw_answer: str
    model: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    fw_l2_result: FWL2Result | None = None

    @property
    def was_redacted(self) -> bool:
        return self.fw_l2_result is not None and self.fw_l2_result.has_pii

    @property
    def injection_detected(self) -> bool:
        return self.fw_l2_result is not None and self.fw_l2_result.injection_detected

    def __str__(self) -> str:
        sections = [c.metadata.get("section", "N/A") for c in self.chunks]
        patients = set(c.metadata.get("patient_name", "N/A") for c in self.chunks)
        redacted_info = ""
        if self.fw_l2_result and self.fw_l2_result.has_pii:
            redacted_info = f"\n  Redacted: {self.fw_l2_result.detection_summary}"
        return (
            f"RAGResponse (model={self.model})\n"
            f"  Query:    {self.query}\n"
            f"  Chunks:   {len(self.chunks)} ({', '.join(sections)})\n"
            f"  Patients: {', '.join(patients)}{redacted_info}\n"
            f"  Answer:   {self.answer}"
        )


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve -> generate -> validate."""

    def __init__(
        self,
        index_dir: str | Path,
        fw_l2: FWL2 | None = None,
        **generator_kwargs,
    ):
        """Initialize the pipeline.

        Args:
            index_dir: Path to FAISS index directory.
            fw_l2: Optional FW-L2 firewall instance. If None, no validation is applied.
            **generator_kwargs: Passed to Generator (model, base_url, system_prompt).
        """
        self.retriever = Retriever(index_dir)
        self.generator = Generator(**generator_kwargs)
        self.fw_l2 = fw_l2

    def query(
        self,
        query: str,
        top_k: int = TOP_K,
        sections: list[str] | None = None,
        temperature: float = 0.1,
    ) -> RAGResponse:
        """Run the full RAG pipeline.

        Args:
            query: User's natural language question.
            top_k: Number of chunks to retrieve.
            sections: Optional section filter for retrieval.
            temperature: LLM sampling temperature.

        Returns:
            RAGResponse with answer and retrieved chunks.
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(query, top_k=top_k, sections=sections)

        # Step 2: Extract text for the LLM context
        context_texts = [chunk.text for chunk in chunks]

        # Step 3: Generate answer
        gen_response = self.generator.generate(
            query=query,
            context_chunks=context_texts,
            temperature=temperature,
        )

        raw_answer = gen_response.answer

        # Step 4: FW-L2 validation (if enabled)
        fw_l2_result = None
        answer = raw_answer
        if self.fw_l2:
            fw_l2_result = self.fw_l2.validate(raw_answer)
            answer = fw_l2_result.sanitized_text_generic

        return RAGResponse(
            query=query,
            answer=answer,
            raw_answer=raw_answer,
            model=gen_response.model,
            chunks=chunks,
            fw_l2_result=fw_l2_result,
        )

    async def query_async(
        self,
        query: str,
        top_k: int = TOP_K,
        sections: list[str] | None = None,
        temperature: float = 0.1,
    ) -> RAGResponse:
        """Async version of query(). CPU-bound steps run in a thread pool."""
        import asyncio

        # Retrieval (embedding + FAISS + CrossEncoder reranking) is CPU-bound.
        # Running it in a thread pool keeps the event loop free to accept other
        # requests while this query waits for the reranker.
        chunks = await asyncio.to_thread(
            self.retriever.retrieve, query, top_k=top_k, sections=sections
        )
        context_texts = [chunk.text for chunk in chunks]

        gen_response = await self.generator.generate_async(
            query=query,
            context_chunks=context_texts,
            temperature=temperature,
        )

        raw_answer = gen_response.answer

        # FW-L2 NER validation is also CPU-bound (BERT inference).
        fw_l2_result = None
        answer = raw_answer
        if self.fw_l2:
            fw_l2_result = await asyncio.to_thread(self.fw_l2.validate, raw_answer)
            answer = fw_l2_result.sanitized_text_generic

        return RAGResponse(
            query=query,
            answer=answer,
            raw_answer=raw_answer,
            model=gen_response.model,
            chunks=chunks,
            fw_l2_result=fw_l2_result,
        )


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parent.parent.parent.parent
    index_dir = project_root / "index"

    if not (index_dir / "faiss.index").exists():
        index_dir = project_root / "data" / "index"

    if not (index_dir / "faiss.index").exists():
        print("Error: No FAISS index found. Run 'uv run ingestion' first.")
        sys.exit(1)

    pipeline = RAGPipeline(index_dir)

    query = "What medications is the patient taking for hypertension?"
    print(f"Query: {query}\n")

    response = pipeline.query(query)
    print(response)