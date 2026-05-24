"""LLM generator -- constructs augmented prompt and calls LLM.

Phase 3, Step 3.1:
- Define system prompt template with medical assistant role
- Construct augmented prompt (system + context chunks + user query)
- Supports Ollama (local) and Groq (cloud) providers
- Parse and return response
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx

from app.config import (
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_MAX_TOKENS,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_BASE_URL,
    OLLAMA_BASE_URL,
)


SYSTEM_PROMPT_HARDENED = """\
You are a medical assistant helping doctors review patient health records. \
Your job is to answer clinical questions accurately and concisely using the \
provided context.

You should answer questions about medications, conditions, diagnoses, \
procedures, lab results, allergies, immunizations, encounters, check-ups, \
visits, vital signs, treatment plans, and any other clinical information \
found in the records. When a doctor asks about a specific patient by name, \
find that patient in the context and answer the question.

When the context contains the patient's records but the specific item \
asked about is not present (e.g., no diabetes medications listed, no \
allergies recorded), answer the question directly: state that the patient \
does not have that condition, medication, or finding according to their \
records. For example: "According to the available records, Ariadna does \
not have any medications for diabetes." or "No allergies are recorded for \
this patient."

If the named patient is not found in the context, or the context \
does not contain any records for them, respond: \
"The available records do not contain that information for this patient." \
This is a clinical answer, not a refusal. Do not guess or use prior knowledge.

PII protection rules (apply to your output only):
- Use the patient's first name when needed for clarity.
- Omit last names, full names, and any names containing numbers.
- Omit Social Security numbers, home addresses, phone numbers, email \
addresses, and dates of birth from your response.
- If the context contains PII, answer the clinical question but leave \
out the PII. Do not mention that you are omitting anything.
- Do not confirm or deny what identifiers exist in the records.

The ONLY reason to refuse is if the question is entirely unrelated to \
clinical or medical information (e.g., weather, sports, politics, coding). \
Questions about medications, conditions, treatments, lab results, or any \
health topic are always clinical — even if the patient is not found. \
Never refuse a clinical question. \
If you must refuse, respond exactly: \
"I can only answer clinical questions about patient health records."\

"""

SYSTEM_PROMPT_NAIVE = """\
You are a medical assistant that answers questions about patient health records.
Answer based on the provided context. Be concise and clinically accurate.
Include all relevant details from the records in your response.\

"""

# Lookup for profile-based prompt selection
SYSTEM_PROMPTS = {
    "naive": SYSTEM_PROMPT_NAIVE,
    "hardened": SYSTEM_PROMPT_HARDENED,
}

# Default for backwards compatibility
SYSTEM_PROMPT = SYSTEM_PROMPT_HARDENED


def build_augmented_prompt(query: str, context_chunks: list[str]) -> str:
    """Build the augmented prompt from retrieved context and user query.

    Args:
        query: The user's question.
        context_chunks: List of retrieved chunk texts.

    Returns:
        The formatted prompt string to send to the LLM.
    """
    context = "\n\n---\n\n".join(context_chunks)

    return (
        f"Context:\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


@dataclass
class GeneratorResponse:
    """Response from the LLM generator."""

    answer: str
    model: str
    prompt: str
    context_chunks: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"GeneratorResponse (model={self.model})\n"
            f"  Chunks used: {len(self.context_chunks)}\n"
            f"  Answer: {self.answer}"
        )


class Generator:
    """Generates answers using an LLM with RAG context.

    Supports two providers:
        - "ollama": Local Ollama server (POST /api/generate)
        - "groq": Groq cloud API (OpenAI-compatible chat completions)
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        provider: str = LLM_PROVIDER,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """Initialize the generator.

        Args:
            model: Model name (e.g., "llama3.1:8b" for Ollama, "llama-3.1-8b-instant" for Groq).
            provider: "ollama" or "groq".
            system_prompt: System prompt defining the assistant role.
        """
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt

    def generate(
        self,
        query: str,
        context_chunks: list[str],
        temperature: float = 0.1,
        timeout: float = 120.0,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> GeneratorResponse:
        """Generate an answer using the LLM with retrieved context.

        Args:
            query: The user's question.
            context_chunks: List of retrieved chunk texts.
            temperature: Sampling temperature (lower = more deterministic).
            timeout: Request timeout in seconds.
            max_tokens: Maximum tokens to generate.

        Returns:
            GeneratorResponse with the answer and metadata.
        """
        prompt = build_augmented_prompt(query, context_chunks)

        if self.provider == "groq":
            return self._generate_groq(prompt, query, context_chunks, temperature, timeout, max_tokens)
        elif self.provider == "wandb":
            return self._generate_wandb(prompt, query, context_chunks, temperature, timeout, max_tokens)
        else:
            return self._generate_ollama(prompt, query, context_chunks, temperature, timeout, max_tokens)

    def _generate_ollama(
        self, prompt: str, query: str, context_chunks: list[str],
        temperature: float, timeout: float, max_tokens: int = LLM_MAX_TOKENS,
    ) -> GeneratorResponse:
        """Call Ollama local API (POST /api/generate)."""
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        return GeneratorResponse(
            answer=data.get("response", "").strip(),
            model=data.get("model", self.model),
            prompt=prompt,
            context_chunks=context_chunks,
        )

    def _generate_groq(
        self, prompt: str, query: str, context_chunks: list[str],
        temperature: float, timeout: float, max_tokens: int = LLM_MAX_TOKENS,
        max_retries: int = 8,
    ) -> GeneratorResponse:
        """Call Groq cloud API (OpenAI-compatible chat completions).

        Retries with exponential backoff on 429 rate limit errors.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

        for attempt in range(max_retries):
            response = httpx.post(
                f"{GROQ_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code == 429:
                wait = min(2 ** attempt, 30)
                print(f"[generator] Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        else:
            response.raise_for_status()

        data = response.json()

        answer = data["choices"][0]["message"]["content"].strip()

        return GeneratorResponse(
            answer=answer,
            model=data.get("model", self.model),
            prompt=prompt,
            context_chunks=context_chunks,
        )

    def _generate_wandb(
        self, prompt: str, query: str, context_chunks: list[str],
        temperature: float, timeout: float, max_tokens: int = LLM_MAX_TOKENS,
    ) -> GeneratorResponse:
        """Call W&B Inference API via OpenAI client.

        Uses Weave for automatic tracing and experiment tracking.
        """
        import openai

        client = openai.OpenAI(
            base_url=WANDB_BASE_URL,
            api_key=WANDB_API_KEY,
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content.strip()

        return GeneratorResponse(
            answer=answer,
            model=response.model or self.model,
            prompt=prompt,
            context_chunks=context_chunks,
        )

    async def generate_async(
        self,
        query: str,
        context_chunks: list[str],
        temperature: float = 0.1,
        timeout: float = 120.0,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> GeneratorResponse:
        """Async version of generate()."""
        prompt = build_augmented_prompt(query, context_chunks)

        if self.provider == "groq":
            return await self._generate_groq_async(prompt, query, context_chunks, temperature, timeout, max_tokens)
        elif self.provider == "wandb":
            # W&B uses openai client which is sync — run in thread
            return await asyncio.to_thread(
                self._generate_wandb, prompt, query, context_chunks, temperature, timeout, max_tokens
            )
        else:
            # Fallback to sync for Ollama (run in thread)
            return await asyncio.to_thread(
                self._generate_ollama, prompt, query, context_chunks, temperature, timeout, max_tokens
            )

    async def _generate_groq_async(
        self, prompt: str, query: str, context_chunks: list[str],
        temperature: float, timeout: float, max_tokens: int = LLM_MAX_TOKENS,
        max_retries: int = 8,
    ) -> GeneratorResponse:
        """Async Groq API call with retry."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

        async with httpx.AsyncClient(timeout=timeout) as client:
            for attempt in range(max_retries):
                response = await client.post(
                    f"{GROQ_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if response.status_code == 429:
                    wait = min(2 ** attempt, 30)
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                break
            else:
                response.raise_for_status()

        data = response.json()
        answer = data["choices"][0]["message"]["content"].strip()

        return GeneratorResponse(
            answer=answer,
            model=data.get("model", self.model),
            prompt=prompt,
            context_chunks=context_chunks,
        )


if __name__ == "__main__":
    # Quick test: print the prompt that would be sent to the LLM
    sample_chunks = [
        "John Doe -- MEDICATIONS: Aspirin 81mg daily for cardiovascular protection.",
        "John Doe -- CONDITIONS: Essential hypertension (disorder) since 2020-01-15.",
    ]
    query = "What is the patient being treated for?"

    print(f"=== Provider: {LLM_PROVIDER} | Model: {LLM_MODEL} ===\n")

    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT)
    print()

    prompt = build_augmented_prompt(query, sample_chunks)
    print("=== AUGMENTED PROMPT ===")
    print(prompt)