"""
Generation layer: takes retrieved (and reranked) chunks + the user's
question, and calls the Groq API to produce a grounded answer.
"""
try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency
    Groq = None

from src.config import config

_client = None

PROMPT_TEMPLATE = """You are a domain expert assistant answering questions using ONLY the provided context.
If the context does not contain enough information to answer, say so explicitly — do not guess.

Context:
{context}

Question: {question}

Answer (grounded strictly in the context above):"""


def get_client():
    global _client
    if _client is None:
        if not config.GROQ_API_KEY or Groq is None:
            return None
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = PROMPT_TEMPLATE.format(context=context_block, question=question)

    client = get_client()
    if client is None:
        return "I don't have a configured LLM backend, so I can only provide a context-based fallback."

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content
