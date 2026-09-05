"""
Consultation summary + summary provenance.

Input : transkrip percakapan + structured health context + evidence yang dibahas
Output : ringkasan terstruktur di mana SETIAP bagian membawa asal-usulnya.

Aturan keras: summary hanya boleh berisi informasi yang benar-benar muncul
dalam percakapan atau berasal dari evidence. Field yang tidak diketahui tetap
`null` / list kosong, tidak diisi tebakan.
"""

import json

from ..context.conversation import public_conversation_id
import logging
from typing import Any, Dict, List, Optional

from ..contracts import (
    PRELIMINARY_ASSESSMENT_DISCLAIMER,
    PRELIMINARY_ASSESSMENT_STATUS,
    EvidenceItem,
    HealthContext,
    Provenance,
)
from ..context.extractor import extract_health_context, health_context_from_dict
from ..evidence.provenance import provenance_entry
from ..safety.validator import detect_emergency, detect_high_risk_population
from .. import runtime

logger = logging.getLogger(__name__)


def _collect_messages(session) -> List[Dict[str, str]]:
    return [
        {"role": m.role, "content": m.content, "intent": m.intent}
        for m in session.messages.all()
    ]


def _evidence_from_messages(session) -> List[Dict[str, Any]]:
    """Kumpulkan evidence yang pernah dipakai selama sesi, tanpa duplikat."""
    seen: set = set()
    collected: List[Dict[str, Any]] = []
    for message in session.messages.all():
        if not message.evidence_refs:
            continue
        try:
            refs = json.loads(message.evidence_refs)
        except (ValueError, TypeError):
            continue
        for ref in refs or []:
            key = (ref.get("doi") or ref.get("source_id") or ref.get("title") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            collected.append({
                "source_id": ref.get("source_id"),
                "title": ref.get("title"),
                "doi": ref.get("doi"),
            })
    return collected


def build_summary(session,
                  extra_evidence: Optional[List[EvidenceItem]] = None) -> Dict[str, Any]:
    """
    Susun ringkasan konsultasi dari sebuah ConversationSession.

    Semua nilai dibungkus `{"value": ..., "provenance": ...}` kecuali field
    metadata, sehingga consumer tahu mana yang dilaporkan user, mana yang
    disimpulkan AI, dan mana yang berbasis bukti.
    """
    messages = _collect_messages(session)
    user_texts = [m["content"] for m in messages if m["role"] == "user"]
    transcript = "\n".join(user_texts)

    # 1) Health context: gabungan snapshot sesi + ekstraksi ulang transkrip.
    context = HealthContext()
    if session.health_context:
        try:
            context = health_context_from_dict(json.loads(session.health_context))
        except (ValueError, TypeError):
            context = HealthContext()
    for text in user_texts:
        context = extract_health_context(text, previous=context)

    # 2) Evidence yang benar-benar dibahas.
    evidence_discussed = _evidence_from_messages(session)
    for item in (extra_evidence or []):
        evidence_discussed.append({
            "source_id": item.source_id or item.chunk_id,
            "title": item.title,
            "doi": item.doi or None,
        })

    # 3) Informasi relevan lain yang dilaporkan user.
    relevant_information: List[Dict[str, Any]] = []
    for field_name, label in (
        ("severity", "Tingkat keparahan"),
        ("onset", "Awitan keluhan"),
        ("progression", "Perkembangan keluhan"),
    ):
        value = getattr(context, field_name, None)
        if value:
            relevant_information.append(
                provenance_entry(f"{label}: {value}", Provenance.USER_REPORTED)
            )
    for medication in context.medications or []:
        relevant_information.append(
            provenance_entry(f"Obat yang disebut: {medication}", Provenance.USER_REPORTED)
        )
    for allergy in context.allergies or []:
        relevant_information.append(
            provenance_entry(f"Alergi: {allergy}", Provenance.USER_REPORTED)
        )
    for history in context.relevant_history or []:
        relevant_information.append(
            provenance_entry(f"Riwayat: {history}", Provenance.USER_REPORTED)
        )

    # 4) Safety notes berdasarkan sinyal nyata di transkrip.
    safety_notes: List[Dict[str, Any]] = []
    emergency = detect_emergency(transcript)
    if emergency:
        safety_notes.append(provenance_entry(
            "Terdapat sinyal kegawatdaruratan pada keluhan yang dilaporkan; "
            "perlu evaluasi medis segera.",
            Provenance.SYSTEM_GENERATED,
            detail=", ".join(sorted(set(emergency))[:5]),
        ))
    risk_population = detect_high_risk_population(transcript)
    if risk_population:
        safety_notes.append(provenance_entry(
            "Termasuk kelompok yang memerlukan kehati-hatian ekstra.",
            Provenance.SYSTEM_GENERATED,
            detail=", ".join(sorted(set(risk_population))[:4]),
        ))
    safety_notes.append(provenance_entry(
        PRELIMINARY_ASSESSMENT_DISCLAIMER, Provenance.SYSTEM_GENERATED
    ))

    # 5) Preliminary assessment, bukan diagnosis.
    preliminary: Optional[Dict[str, Any]] = None
    if context.symptoms:
        descriptor = ", ".join(context.symptoms[:5])
        duration_part = f" selama {context.duration}" if context.duration else ""
        preliminary = provenance_entry(
            f"Pasien melaporkan {descriptor}{duration_part}. "
            "Belum dapat ditentukan penyebab pastinya dari percakapan ini.",
            Provenance.AI_INFERRED if not evidence_discussed else Provenance.EVIDENCE_SUPPORTED,
        )

    # 6) Langkah lanjutan.
    next_steps: List[Dict[str, Any]] = []
    if emergency:
        next_steps.append(provenance_entry(
            "Segera cari pertolongan gawat darurat (119 / UGD terdekat).",
            Provenance.SYSTEM_GENERATED,
        ))
    if context.symptoms:
        if not context.duration:
            next_steps.append(provenance_entry(
                "Lengkapi informasi durasi keluhan saat konsultasi.",
                Provenance.SYSTEM_GENERATED,
            ))
        next_steps.append(provenance_entry(
            "Konsultasikan keluhan dengan tenaga kesehatan untuk pemeriksaan langsung.",
            Provenance.SYSTEM_GENERATED,
        ))
    if not next_steps:
        next_steps.append(provenance_entry(
            "Tidak ada langkah spesifik yang dapat disimpulkan dari percakapan ini.",
            Provenance.SYSTEM_GENERATED,
        ))

    summary: Dict[str, Any] = {
        "session_id": public_conversation_id(session),
        "consumer": session.consumer,
        "status": PRELIMINARY_ASSESSMENT_STATUS,
        "is_diagnosis": False,
        "chief_complaint": (
            provenance_entry(context.chief_complaint, Provenance.USER_REPORTED)
            if context.chief_complaint else None
        ),
        "symptoms": [
            provenance_entry(s, Provenance.USER_REPORTED) for s in (context.symptoms or [])
        ],
        "duration": (
            provenance_entry(context.duration, Provenance.USER_REPORTED)
            if context.duration else None
        ),
        "relevant_information": relevant_information,
        "preliminary_assessment": preliminary,
        "evidence_discussed": evidence_discussed,
        "recommended_next_step": next_steps,
        "safety_notes": safety_notes,
        "health_context": context.to_dict(),
        "metadata": {
            "turns": len(messages),
            "user_messages": len(user_texts),
            "generated_from": "conversation_transcript",
        },
    }
    return summary


def persist_summary(session, summary: Dict[str, Any]):
    """Simpan summary ke DB. Kegagalan tidak boleh menggagalkan response."""
    try:
        ConsultationSummary = runtime.model("ConsultationSummary")
        chief = summary.get("chief_complaint") or {}
        return ConsultationSummary.objects.create(
            session=session,
            chief_complaint=(chief.get("value") or "") if isinstance(chief, dict) else "",
            payload=json.dumps(summary, ensure_ascii=False),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("[SUMMARY] gagal menyimpan summary: %s", exc)
        return None
