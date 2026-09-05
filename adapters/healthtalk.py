"""
Adapter response untuk consumer eksternal (§13, §21).

    Internal Response
           |
     +-----+------+
     v            v
  Existing     HealthTalk
  API Format   API Format

HealthTalk hanya melihat bentuk di bawah ini; ia tidak perlu tahu apa pun
tentang implementasi internal Healthify.
"""

from typing import Any, Dict

from ..contracts import IntelligenceResponse

from ..citations import strip_citation_markers


def to_consumer_response(response: IntelligenceResponse,
                         include_evidence: bool = True,
                         include_sources: bool = True) -> Dict[str, Any]:
    """Bentuk response publik untuk endpoint /api/v1/intelligence/query."""
    data: Dict[str, Any] = {
        # Tanpa penanda sitasi; yang membutuhkan pemetaan kalimat ke bukti
        # memakai `answer_annotated`.
        "answer": strip_citation_markers(response.answer),
        "answer_annotated": response.answer,
        "intent": response.intent.value,
        "mode": response.mode.value,
        "conversation_id": response.conversation_id,
        "health_context": response.health_context.to_dict(),
        "evidence": (
            [item.to_public_dict() for item in response.evidence]
            if include_evidence else []
        ),
        "claims": [claim.to_dict() for claim in response.claims],
        "evidence_status": response.evidence_status.value,
        "uncertainty": response.uncertainty,
        "safety": {
            "decision": response.safety_decision.value,
            "flags": [flag.to_dict() for flag in response.safety_flags],
        },
        "safety_flags": [flag.to_dict() for flag in response.safety_flags],
        "preliminary_assessment": response.preliminary_assessment,
        "metadata": response.metadata,
    }

    if not include_sources:
        for item in data["evidence"]:
            item.pop("doi", None)
            item.pop("url", None)

    return data


def to_simple_response(response: IntelligenceResponse) -> Dict[str, Any]:
    """
    Bentuk ringkas untuk consumer yang hanya butuh **informasi kesehatan
    beserta sumber jurnalnya**.

    Tanpa label verifikasi, intent, maupun status internal: itu urusan produk
    Healthify, bukan sesuatu yang harus ditafsirkan consumer.

    Field:
        answer          teks jawaban siap tampil
        sources         jurnal pendukung; `url` sudah dipastikan hidup
        has_evidence    False berarti bukti tidak memadai dan `answer` berisi
                        pernyataan jujur soal itu
        notice          peringatan yang wajib ditampilkan bila terisi
        conversation_id kirim balik agar konteks percakapan tersambung
        sources_reused  True bila rujukannya sama dengan giliran sebelumnya
        request_id      untuk pelaporan masalah
    """
    critical = [
        flag for flag in response.safety_flags
        if getattr(flag, "severity", "") == "critical"
    ]
    notice = critical[0].message if critical else None

    answer = strip_citation_markers(response.answer)

    return {
        "answer": answer,
        "sources": [
            {
                "title": item.title,
                "url": item.url or None,
                "doi": item.doi or None,
                "publisher": item.publisher or None,
                "year": item.published_year,
                "relevance": round(float(item.relevance), 3),
                "snippet": (item.snippet or "")[:400],
            }
            for item in response.evidence
        ],
        "has_evidence": bool(response.evidence),
        "notice": notice,
        "conversation_id": response.conversation_id,
        "sources_reused": (response.metadata or {}).get("evidence_source") == "conversation",
        "request_id": (response.metadata or {}).get("request_id"),
    }
