"""
reflective_agent.py

Wraps MultiModalEngine with a reflect-and-retry loop.

Instead of blindly answering with whatever Qdrant returns, this agent
pauses after retrieval, grades the context quality with a fast LLM call,
and retries with a rewritten query if the score is too low.

Flow:
  1.  retrieve from Qdrant  (same as before)
  2.  LLM grades the context  1-5
  3.  score < threshold  →  rewrite query, go back to step 1
  4.  score >= threshold OR max retries hit  →  answer with the best
      context seen across all attempts
"""

import json
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

import config.config as config
from src.core.rag_engine import MultiModalEngine, QueryResult


# ── tuneable knobs ─────────────────────────────────────────────────────────────

# Score at which we consider the context "good enough" and stop retrying.
# 3 = partially useful.  Bump to 4 if you're getting too many noisy answers.
REFLECTION_THRESHOLD = 3

# How many extra retrieval attempts to allow after the first one.
MAX_RETRIES = 2

# How many characters of context to send to the grader.
# Grading doesn't need the full text — 3 000 chars is more than enough.
CONTEXT_PREVIEW_LENGTH = 3000


# ── prompts ────────────────────────────────────────────────────────────────────

_REFLECT_SYSTEM = (
    "You are a retrieval quality evaluator. "
    "You receive a question and some retrieved context, then score how well "
    "the context covers the question. "
    "Respond with valid JSON only — no explanation, no markdown fences."
)

_REFLECT_PROMPT = """\
Question:
{question}

Retrieved context:
\"\"\"
{context}
\"\"\"

Score the context quality from 1 to 5:
  1 — completely irrelevant
  2 — mostly off-topic, maybe one tangential piece
  3 — partially useful, addresses some aspects but missing key info
  4 — mostly sufficient, only minor gaps
  5 — fully and clearly answers the question

Rules:
- Score based on the ORIGINAL question, not the retrieval query.
- If score < 4, suggest a better search query in "rewrite".
- If score >= 4, leave "rewrite" as an empty string.

Respond ONLY as JSON, nothing else:
{{"score": <int 1-5>, "reason": "<one sentence>", "rewrite": "<better query or empty string>"}}"""


# ── internal dataclass ─────────────────────────────────────────────────────────

@dataclass
class _Reflection:
    score:          int
    reason:         str
    rewrite:        str   # empty = "don't retry"
    attempt:        int
    context_chars:  int   # just for logging


# ── agent ──────────────────────────────────────────────────────────────────────

class ReflectiveRAGAgent:
    """
    Drop-in replacement for engine.ask_question() with a self-grading loop.

    Basic usage
    -----------
        from engine import MultiModalEngine
        from reflective_agent import ReflectiveRAGAgent

        engine = MultiModalEngine()
        agent  = ReflectiveRAGAgent(engine)
        result = agent.query("What does figure 3 show about latency?")
        print(result.answer)

    Config hook
    -----------
    Add NVIDIA_REFLECTION_MODEL to config.py to use a smaller/cheaper model
    for grading calls (recommended — saves cost and latency).
    Example:  NVIDIA_REFLECTION_MODEL = "meta/llama-3.1-8b-instruct"
    Falls back to NVIDIA_LLM_MODEL if not set.
    """

    def __init__(self, engine: MultiModalEngine, threshold: int = REFLECTION_THRESHOLD):
        self.engine    = engine
        self.threshold = threshold

        # Reuse the same NVIDIA base URL so no extra auth is needed.
        # We intentionally keep this client separate from the main engine
        # so we can swap to a cheaper model just for grading.
        if getattr(config, "LLM", None) == "NVIDIA":
            self._client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=getattr(config, "NVIDIA_API_KEY", ""),
            )
            self._model = getattr(
                config,
                "NVIDIA_REFLECTION_MODEL",   # cheaper model if configured
                config.NVIDIA_LLM_MODEL,     # fallback to main model
            )
        else:
            # OpenAI path — engine.mm_llm handles the call inside _grade_context
            self._client = None
            self._model  = "gpt-4o-mini"

    # ── public ────────────────────────────────────────────────────────────────

    def query(
        self,
        user_query:  str,
        temperature: float = 0.7,
        max_tokens:  int   = 500,
    ) -> QueryResult:
        """
        Run retrieval, grade the result, retry if needed, then answer.
        Always answers the original question regardless of how the query
        was rewritten during retries.
        """
        current_query    = user_query
        best_text_nodes  = []
        best_image_nodes = []
        best_score       = 0

        # attempt 1 is the initial retrieval, attempts 2..N are retries
        for attempt in range(1, MAX_RETRIES + 2):
            print(f"\n[reflect] Attempt {attempt}/{MAX_RETRIES + 1} — '{current_query[:80]}'")

            text_nodes, image_nodes = self.engine.retrieve_documents(current_query)

            if not text_nodes:
                print("[reflect] Retrieved nothing — stopping early.")
                break

            context_str = self._build_context_str(text_nodes)
            grade       = self._grade_context(user_query, context_str, attempt)

            print(f"[reflect] Score {grade.score}/5 — {grade.reason}")

            # always keep the best result we've seen so far
            if grade.score > best_score:
                best_score       = grade.score
                best_text_nodes  = text_nodes
                best_image_nodes = image_nodes

            # good enough — no need to retry
            if grade.score >= self.threshold:
                print(f"[reflect] Threshold met (score {grade.score} >= {self.threshold}).")
                break

            # we've used all retries
            if attempt == MAX_RETRIES + 1:
                print("[reflect] Max retries reached — using best context found.")
                break

            # the grader didn't give us a rewrite suggestion
            if not grade.rewrite.strip():
                print("[reflect] No rewrite suggested — stopping.")
                break

            print(f"[reflect] Retrying with rewritten query: '{grade.rewrite}'")
            current_query = grade.rewrite
            time.sleep(0.15)   # tiny pause so we don't hammer the API

        # ── nothing useful was found at all ───────────────────────────────────
        if not best_text_nodes:
            return QueryResult(
                answer=(
                    "I couldn't find relevant information in the knowledge base "
                    "for your question. Try rephrasing or uploading more documents."
                )
            )

        print(
            f"[reflect] Generating final answer "
            f"(best score={best_score}, "
            f"text_nodes={len(best_text_nodes)}, "
            f"image_nodes={len(best_image_nodes)})."
        )

        # always answer the original question, not the rewritten one
        return self.engine.generate_rag_response(
            prompt=user_query,
            final_text_nodes=best_text_nodes,
            final_image_nodes=best_image_nodes,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    def _grade_context(self, question: str, context_str: str, attempt: int) -> _Reflection:
        """
        Ask the LLM to score how well context_str answers question.
        Returns a safe default (score=3, no rewrite) on any failure so the
        main loop can keep running without crashing.
        """
        prompt = _REFLECT_PROMPT.format(
            question=question,
            context=context_str[:CONTEXT_PREVIEW_LENGTH],
        )

        raw = ""
        try:
            if self._client:
                # NVIDIA / OpenAI-compatible path
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _REFLECT_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.0,   # scoring should be deterministic
                    max_tokens=150,
                )
                raw = response.choices[0].message.content.strip()
            else:
                # mm_llm (OpenAI local) path — no system message support here
                raw = str(
                    self.engine.mm_llm.complete(
                        prompt=f"{_REFLECT_SYSTEM}\n\n{prompt}",
                        image_documents=[],
                    )
                ).strip()

            # models sometimes wrap JSON in ``` fences despite being told not to
            raw = self._strip_fences(raw)
            parsed = json.loads(raw)

            return _Reflection(
                score=int(parsed.get("score", 3)),
                reason=str(parsed.get("reason", "")),
                rewrite=str(parsed.get("rewrite", "")),
                attempt=attempt,
                context_chars=len(context_str),
            )

        except json.JSONDecodeError:
            print(f"[reflect] Could not parse grading response: {raw!r}")
        except Exception as exc:
            print(f"[reflect] Grading call failed on attempt {attempt}: {exc}")

        # safe fallback — "partially useful, don't retry"
        return _Reflection(
            score=3,
            reason="grading unavailable",
            rewrite="",
            attempt=attempt,
            context_chars=len(context_str),
        )

    def _build_context_str(self, text_nodes) -> str:
        return "\n\n---\n\n".join(
            node.node.get_content() for node in text_nodes
        )

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` wrappers if present."""
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        return "\n".join(inner).strip()