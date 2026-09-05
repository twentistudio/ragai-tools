"""
Re-ranking evidence.

Implementasinya berada di `api.intelligence.evidence.quality` karena skor
ranking dan skor kualitas memakai dimensi yang sama (semantic_relevance,
source_quality, publication_recency, evidence_type, context_match).

Modul ini mengekspos ulang API-nya agar susunan paket sesuai peta arsitektur
dan bobot/heuristik bisa diganti tanpa menyentuh pipeline retrieval.
"""

from ..evidence.quality import (  # noqa: F401
    DEFAULT_WEIGHTS,
    compute_evidence_score,
    rank_evidence,
    score_context_match,
    score_evidence_type,
    score_recency,
    score_source_quality,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "compute_evidence_score",
    "rank_evidence",
    "score_context_match",
    "score_evidence_type",
    "score_recency",
    "score_source_quality",
]
