"""
Claim verification di atas shared evidence layer (§11).

Claim engine Healthify yang lama TIDAK diganti: `/api/verify/` tetap memakai
`api.ai_adapter.call_ai_verify` seperti sebelumnya. Modul ini adalah jalur
claim verification *di dalam* Health Intelligence Engine, sehingga HealthTalk
(atau consumer lain) bisa memverifikasi klaim lewat kontrak yang sama dengan
konsultasi, dan berbagi evidence layer yang sama.

    Claim -> Evidence Retrieval -> Evidence Evaluation
          -> Supported / Unsupported / Inconclusive -> Explanation -> Sources
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..contracts import EvidenceItem, EvidenceStatus
from ..reasoning import llm

logger = logging.getLogger(__name__)

VERDICT_SUPPORTED = "supported"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_INCONCLUSIVE = "inconclusive"

# Pemetaan ke label publik Healthify yang sudah ada (jangan diubah).
VERDICT_TO_LEGACY_LABEL = {
    VERDICT_SUPPORTED: "valid",
    VERDICT_UNSUPPORTED: "hoax",
    VERDICT_INCONCLUSIVE: "uncertain",
}

_SUPPORT_CUES = [
    r"\b(terbukti|menunjukkan bahwa|berhubungan dengan|berkaitan dengan|efektif|bermanfaat)\b",
    r"\b(associated with|is effective|demonstrated|evidence supports|significantly reduced)\b",
]
_REFUTE_CUES = [
    r"\b(tidak terbukti|tidak ada bukti|tidak efektif|tidak berhubungan|membantah|keliru|tidak didukung)\b",
    r"\b(no evidence|not effective|no association|refut|contradict|debunk|myth)\w*\b",
]

_EVAL_SYSTEM = (
    "Anda adalah evaluator klaim kesehatan berbasis bukti. Nilai klaim HANYA "
    "berdasarkan EVIDENCE yang diberikan. Dilarang memakai pengetahuan di luar "
    "EVIDENCE, dan dilarang menyebut sumber/DOI/URL yang tidak ada di EVIDENCE. "
    "Jika EVIDENCE tidak membahas hubungan yang diklaim, jawab 'inconclusive'."
)


@dataclass
class ClaimEvaluation:
    """Satu pernyataan dalam jawaban beserta bukti yang menopangnya."""

    verdict: str = VERDICT_INCONCLUSIVE
    confidence: Optional[float] = None
    explanation: str = ""
    supporting_evidence_ids: List[str] = field(default_factory=list)
    method: str = "heuristic"

    def legacy_label(self) -> str:
        return VERDICT_TO_LEGACY_LABEL.get(self.verdict, "uncertain")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "legacy_label": self.legacy_label(),
            "confidence": self.confidence,
            "explanation": self.explanation,
            "supporting_evidence_ids": self.supporting_evidence_ids,
            "method": self.method,
        }


def _build_eval_prompt(claim: str, evidence: List[EvidenceItem]) -> str:
    blocks = []
    for idx, item in enumerate(evidence, start=1):
        snippet = (item.snippet or "").replace("\n", " ")[:900]
        blocks.append(f"[E{idx}] {item.title or '(tanpa judul)'}\n{snippet}")
    evidence_text = "\n\n".join(blocks)

    return (
        f"KLAIM YANG DINILAI:\n{claim}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Balas HANYA dengan JSON valid berikut (tanpa markdown):\n"
        "{\n"
        '  "verdict": "supported" | "unsupported" | "inconclusive",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "explanation": "penjelasan 2-4 kalimat dalam Bahasa Indonesia, '
        'sebutkan penanda [E1]/[E2] untuk evidence yang dipakai",\n'
        '  "evidence_used": [1, 2]\n'
        "}\n"
        "Aturan: 'supported' hanya bila evidence secara langsung mendukung hubungan "
        "yang diklaim; 'unsupported' bila evidence secara langsung membantahnya; "
        "selain itu 'inconclusive'."
    )


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except (ValueError, TypeError):
            return None
    return None


def _heuristic_evaluation(claim: str, evidence: List[EvidenceItem]) -> ClaimEvaluation:
    """
    Evaluasi tanpa LLM. Konservatif: hanya menyimpulkan bila sinyal jelas,
    selain itu `inconclusive`.
    """
    support_hits = 0
    refute_hits = 0
    used: List[str] = []

    for item in evidence:
        haystack = f"{item.title} {item.snippet}".lower()
        has_support = any(re.search(p, haystack, re.IGNORECASE) for p in _SUPPORT_CUES)
        has_refute = any(re.search(p, haystack, re.IGNORECASE) for p in _REFUTE_CUES)
        if has_support or has_refute:
            used.append(item.source_id or item.chunk_id)
        support_hits += int(has_support)
        refute_hits += int(has_refute)

    top_relevance = evidence[0].relevance if evidence else 0.0

    if refute_hits > support_hits and refute_hits >= 1:
        verdict, confidence = VERDICT_UNSUPPORTED, min(0.75, 0.4 + 0.1 * refute_hits)
    elif support_hits > refute_hits and support_hits >= 2 and top_relevance >= 0.55:
        verdict, confidence = VERDICT_SUPPORTED, min(0.8, 0.45 + 0.1 * support_hits)
    else:
        verdict, confidence = VERDICT_INCONCLUSIVE, None

    explanation = {
        VERDICT_SUPPORTED: (
            "Bukti yang tersedia di basis pengetahuan cenderung mendukung klaim ini. "
            "Penilaian disusun langsung dari cuplikan sumber, tanpa penafsiran tambahan."
        ),
        VERDICT_UNSUPPORTED: (
            "Bukti yang tersedia cenderung tidak mendukung klaim ini. "
            "Penilaian disusun langsung dari cuplikan sumber, tanpa penafsiran tambahan."
        ),
        VERDICT_INCONCLUSIVE: (
            "Bukti yang tersedia belum cukup untuk menyimpulkan klaim ini secara pasti."
        ),
    }[verdict]

    return ClaimEvaluation(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        supporting_evidence_ids=used[:5],
        method="heuristic",
    )


def evaluate_claim(claim: str,
                   evidence: List[EvidenceItem],
                   evidence_status: EvidenceStatus) -> ClaimEvaluation:
    """
    Nilai sebuah klaim terhadap evidence yang sudah divalidasi.

    Bila bukti tidak cukup, hasilnya `inconclusive` — LLM tidak dipanggil (§16).
    """
    claim = (claim or "").strip()
    if not claim or not evidence or evidence_status == EvidenceStatus.INSUFFICIENT_EVIDENCE:
        return ClaimEvaluation(
            verdict=VERDICT_INCONCLUSIVE,
            confidence=None,
            explanation=(
                "Tidak ditemukan bukti yang cukup relevan di basis pengetahuan untuk "
                "menilai klaim ini. Sistem tidak menebak ketika bukti tidak memadai."
            ),
            method="insufficient_evidence",
        )

    raw = llm.generate(
        _build_eval_prompt(claim, evidence),
        system=_EVAL_SYSTEM,
        temperature=0.1,
        max_tokens=600,
    )
    parsed = _parse_llm_json(raw) if raw else None

    if not parsed:
        return _heuristic_evaluation(claim, evidence)

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in (VERDICT_SUPPORTED, VERDICT_UNSUPPORTED, VERDICT_INCONCLUSIVE):
        verdict = VERDICT_INCONCLUSIVE

    confidence: Optional[float]
    try:
        confidence = float(parsed.get("confidence"))
        if confidence > 1.0:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = None
    if verdict == VERDICT_INCONCLUSIVE:
        confidence = None

    used_ids: List[str] = []
    for index in parsed.get("evidence_used") or []:
        try:
            pos = int(index)
        except (TypeError, ValueError):
            continue
        if 1 <= pos <= len(evidence):
            item = evidence[pos - 1]
            used_ids.append(item.source_id or item.chunk_id)

    explanation = str(parsed.get("explanation") or "").strip()
    if not explanation:
        return _heuristic_evaluation(claim, evidence)

    # Bersihkan referensi karangan di penjelasan.
    from ..reasoning.generator import _strip_fabricated_references
    explanation = _strip_fabricated_references(explanation, evidence)

    if evidence_status == EvidenceStatus.PARTIAL and verdict != VERDICT_INCONCLUSIVE:
        if confidence is not None:
            confidence = min(confidence, 0.7)
        explanation += (
            " Bukti yang tersedia terbatas, sehingga penilaian ini belum final."
        )

    return ClaimEvaluation(
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        supporting_evidence_ids=used_ids[:5],
        method="llm",
    )
