"""RAG question-answering chain.

Orchestrates the full retrieve-then-generate pipeline: takes a natural
language query, retrieves relevant document chunks, and generates a
grounded answer using an LLM.
"""

from __future__ import annotations

import logging

from src.rag.llm_client import LLMClient
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are an expert energy market analyst. Answer the question based ONLY on the provided context from energy trading books, research reports, and market publications.

If the context does not contain enough information to answer confidently, say so explicitly rather than speculating.

Context:
{context}

Question: {question}

Instructions:
- Be specific and cite source documents where relevant
- Quantify uncertainty using phrases like "likely", "possibly", "the evidence suggests"
- Keep the answer focused and under 400 words unless a longer answer is required
- If relevant, include numerical data points from the context

Answer:"""


class QAChain:
    """Retrieval-Augmented Generation (RAG) QA chain.

    Combines document retrieval with LLM generation to answer questions
    grounded in the energy trading knowledge base.

    Attributes:
        retriever: Document retriever.
        llm: LLM client for generation.
        prompt_template: Template string for constructing LLM prompts.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
    ) -> None:
        """Initialise the QA chain.

        Args:
            retriever: Configured ``Retriever`` instance.
            llm: Configured ``LLMClient`` instance.
            prompt_template: Prompt template with ``{context}`` and
                ``{question}`` placeholders.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        logger.info("QAChain initialised")

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        return_sources: bool = True,
    ) -> dict:
        """Answer a question using the RAG pipeline.

        Args:
            question: Natural language question.
            top_k: Number of chunks to retrieve.
            return_sources: Whether to include source chunks in the response.

        Returns:
            Dictionary with keys:
            - ``answer``: Generated answer string.
            - ``sources``: List of retrieved source chunks (if requested).
            - ``question``: Original question.
        """
        logger.info("QAChain.ask: '%s'", question[:100])

        # Retrieve relevant context
        chunks = self.retriever.retrieve(question, top_k=top_k)
        if not chunks:
            logger.warning("No context retrieved — answering without context")
            context = "No relevant context found in the knowledge base."
        else:
            context = self.retriever.format_context(chunks)

        # Construct prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question,
        )

        # Generate answer
        try:
            answer = self.llm.complete(prompt)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            answer = f"Error generating answer: {exc}"

        result: dict = {"question": question, "answer": answer}
        if return_sources:
            result["sources"] = chunks

        return result

    def ask_batch(self, questions: list[str]) -> list[dict]:
        """Answer a batch of questions.

        Args:
            questions: List of question strings.

        Returns:
            List of result dictionaries (one per question).
        """
        logger.info("Processing batch of %d questions", len(questions))
        return [self.ask(q) for q in questions]
