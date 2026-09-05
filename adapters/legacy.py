"""
Compatibility adapter untuk produk Healthify yang sudah ada.

    Existing Healthify Request  ->  Compatibility Adapter  ->  Unified Internal Request
    Internal Response           ->  Compatibility Adapter  ->  Existing API Format

Adapter ini memastikan kontrak publik `/api/verify/` (dan turunannya) tidak
berubah walaupun engine internal bertambah kemampuannya.
"""

from typing import Any, Dict, List, Optional

from ..claims.evaluator import VERDICT_TO_LEGACY_LABEL
from ..contracts import EvidenceItem, EvidenceStatus, IntelligenceResponse, Mode


def claim_request_to_payload(claim_text: str,
                             max_evidence: int = 5) -> Dict[str, Any]:
    """Request Healthify lama (hanya `text`) -> unified internal request."""
    return {
        "query": claim_text or "",
        "mode": Mode.CLAIM.value,
        "context": {},
        "options": {
            "include_evidence": True,
            "include_sources": True,
            "max_evidence": max_evidence,
            "language": "id",
        },
    }


def evidence_to_legacy_source(item: EvidenceItem, rank: int = 0) -> Dict[str, Any]:
    """EvidenceItem -> dict sumber sesuai bentuk yang dipakai views/_process_sources."""
    return {
        "title": item.title or "Unknown",
        "doi": item.doi or "",
        "url": item.url or "",
        "authors": item.authors or "",
        "publisher": item.publisher or "",
        "published_date": None,
        "source_type": item.source_type or "journal",
        "relevance_score": float(item.relevance or 0.0),
        "credibility_score": float(item.source_quality or 0.5),
        "excerpt": (item.snippet or "")[:500],
        # metadata tambahan (diabaikan oleh kode lama, berguna untuk audit)
        "_origin": item.origin.value if hasattr(item.origin, "value") else str(item.origin),
        "_doi_verified": bool(item.doi_verified),
        "_link_status": item.link_status,
        "_rank": rank,
    }


def engine_response_to_legacy(response: IntelligenceResponse) -> Dict[str, Any]:
    """
    Internal response -> bentuk yang sama persis dengan keluaran
    `api.ai_adapter.call_ai_verify` (label / confidence / summary / sources).
    """
    evaluation = (response.metadata or {}).get("claim_evaluation") or {}
    verdict = evaluation.get("verdict")

    if verdict in VERDICT_TO_LEGACY_LABEL:
        label = VERDICT_TO_LEGACY_LABEL[verdict]
        confidence = evaluation.get("confidence")
    else:
        label, confidence = "unverified", None

    # Tanpa sumber jurnal terverifikasi, Healthify memang menandai UNVERIFIED.
    has_journal = any(
        (item.doi or item.source_type == "journal") for item in response.evidence
    )
    if response.evidence_status == EvidenceStatus.INSUFFICIENT_EVIDENCE or not has_journal:
        label, confidence = "unverified", None

    return {
        "label": label,
        "confidence": confidence,
        "summary": response.answer or "",
        "sources": [
            evidence_to_legacy_source(item, rank=idx + 1)
            for idx, item in enumerate(response.evidence)
        ],
        "_original_label": verdict or "",
        "_engine": "health_intelligence_engine",
        "_debug": {
            "intent": response.intent.value,
            "evidence_status": response.evidence_status.value,
            "safety_decision": response.safety_decision.value,
            "source_count": len(response.evidence),
        },
    }
