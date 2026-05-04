"""POST /test -- configurable evaluation endpoint.

Like /query but lets you switch profiles (system prompt + FW-L2 config)
and see raw vs sanitized output. Useful for testing different pipeline
configurations without restarting the server.
"""

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.config import INDEX_DIR
from app.models.schemas import TestRequest, TestResponse

# Valid profiles (from weave_eval.py PROFILE_CONFIG)
VALID_PROFILES = {
    "naive", "naive_fw_l2_base", "naive_fw_l2_bert",
    "hardened", "hardened_fw_l2_base", "hardened_fw_l2_bert",
    "baseline", "fw_l2_base", "fw_l2_bert",
}

router = APIRouter()


@router.post("/test", response_model=TestResponse)
async def handle_test(body: TestRequest, request: Request):
    """Run a query with a configurable profile.

    Profiles control the system prompt and FW-L2 backend:
    - naive: no guardrails, no FW-L2
    - naive_fw_l2_base: no guardrails, FW-L2 with spaCy
    - naive_fw_l2_bert: no guardrails, FW-L2 with BERT NER
    - hardened: hardened system prompt, no FW-L2
    - hardened_fw_l2_base: hardened prompt + FW-L2 spaCy
    - hardened_fw_l2_bert: hardened prompt + FW-L2 BERT NER (default)
    """
    from pathlib import Path

    from app.firewall.fw_l2 import FWL2
    from app.rag.generator import SYSTEM_PROMPTS, SYSTEM_PROMPT_HARDENED
    from app.rag.pipeline import RAGPipeline

    if body.profile not in VALID_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile '{body.profile}'. Valid profiles: {sorted(VALID_PROFILES)}",
        )

    # Profile configuration
    from app.evaluation.weave_eval import PROFILE_CONFIG
    config = PROFILE_CONFIG.get(body.profile, PROFILE_CONFIG["hardened_fw_l2_bert"])

    system_prompt = SYSTEM_PROMPTS.get(config["prompt"], SYSTEM_PROMPT_HARDENED)
    ner_backend = config.get("ner_backend")
    fw_l2 = FWL2(ner_backend=ner_backend) if config["fw_l2"] else None

    # Build a pipeline with the requested profile
    index_dir = Path(INDEX_DIR)
    if not (index_dir / "faiss.index").exists():
        raise HTTPException(
            status_code=503,
            detail="FAISS index not built yet. Call POST /ingest first.",
        )

    try:
        pipeline = RAGPipeline(index_dir=index_dir, fw_l2=fw_l2, system_prompt=system_prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not load pipeline: {e}")

    try:
        result = await pipeline.query_async(
            query=body.query,
            top_k=body.top_k,
            sections=body.sections,
            temperature=body.temperature,
        )
    except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
        raise HTTPException(status_code=502, detail=f"LLM provider unreachable: {e}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM request timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"LLM provider returned {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error ({type(e).__name__}): {e}")

    redacted_entities = []
    if result.fw_l2_result and result.fw_l2_result.detections:
        redacted_entities = [d.entity_type for d in result.fw_l2_result.detections]

    sources = list({c.metadata.get("source_file", "") for c in result.chunks})
    sections_retrieved = [c.metadata.get("section", "") for c in result.chunks]

    return TestResponse(
        response=result.answer,
        raw_response=result.raw_answer,
        profile=body.profile,
        model=result.model,
        redacted_entities=redacted_entities,
        was_redacted=result.was_redacted,
        injection_detected=result.injection_detected,
        sources=sources,
        sections_retrieved=sections_retrieved,
        fw_l2_passed=not result.was_redacted and not result.injection_detected,
    )