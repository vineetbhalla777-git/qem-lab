"""
chat_service.py
================
The RAG chatbot: retrieves relevant knowledge-base passages for a user's
question, then either (a) asks an LLM to compose a grounded answer citing
those passages, if ANTHROPIC_API_KEY is configured, or (b) returns a
clearly-labeled extractive answer built directly from the retrieved
passages if no LLM is available. Mode (b) is a genuinely working RAG
system on its own -- just without natural-language composition.
"""

from typing import Dict, List, Optional

from backend import llm_client
from backend.rag.retriever import get_retriever

SYSTEM_PROMPT = """You are the assistant embedded in QEM Lab, a website about \
quantum error mitigation (QEM) built on Qiskit, Qiskit Aer, and Mitiq. \
You help visitors understand quantum computing, quantum errors and noise, \
error mitigation techniques, and the AI/ML components of this project \
(a technique-recommendation model and this RAG chatbot itself).

You are given retrieved passages from the project's own knowledge base \
below. Ground your answer in them when they're relevant, and say so \
naturally -- you don't need to hedge if the passages clearly answer the \
question. You may also use your own general knowledge of quantum \
computing and machine learning to answer questions the knowledge base \
doesn't cover, but don't contradict the retrieved passages if they're on \
topic. Keep answers concise, clear, and plain-spoken -- explain like you \
would to a curious engineer who isn't a quantum physicist. Do not invent \
specific numbers or claims about this specific project that aren't in \
the retrieved passages."""


def _format_context(passages: List[dict]) -> str:
    if not passages:
        return "(No closely related passages were found in the knowledge base.)"
    blocks = []
    for p in passages:
        blocks.append(f"[Source: {p['source']} — {p['title']}]\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


def _extractive_fallback(question: str, passages: List[dict]) -> str:
    if not passages:
        return (
            "I couldn't find anything closely related to that in my knowledge base "
            "(NISQ fundamentals, IBM Quantum Runtime mitigation options, this project's "
            "six techniques, and basic AI/ML concepts). Try rephrasing, or ask about one "
            "of those topics directly.\n\n"
            "(Note: no LLM API key is configured, so I'm answering with direct retrieval "
            "only, not generated prose. Set ANTHROPIC_API_KEY to enable full conversational answers.)"
        )
    lines = [
        "Here's what I found most relevant in the knowledge base "
        "(retrieval-only mode — set ANTHROPIC_API_KEY for a conversational answer):\n"
    ]
    for p in passages:
        lines.append(f"**{p['title']}** _(from {p['source']})_")
        snippet = p["text"]
        if len(snippet) > 500:
            snippet = snippet[:500].rsplit(" ", 1)[0] + "…"
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


def answer_question(question: str, history: Optional[List[Dict[str, str]]] = None) -> dict:
    retriever = get_retriever()
    passages = retriever.search(question, top_k=4)

    llm_answer = None
    if llm_client.is_available():
        context = _format_context(passages)
        history_text = ""
        if history:
            history_text = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:]) + "\n\n"
        user_message = (
            f"{history_text}Retrieved context:\n{context}\n\n"
            f"User question: {question}"
        )
        llm_answer = llm_client.generate(SYSTEM_PROMPT, user_message)

    used_llm = llm_answer is not None
    answer = llm_answer if used_llm else _extractive_fallback(question, passages)

    return {
        "answer": answer,
        "used_llm": used_llm,
        "sources": [{"source": p["source"], "title": p["title"]} for p in passages],
    }
