"""
Health safety layer.

Dijalankan SETELAH response dihasilkan dan SEBELUM dikirim keluar.

Tujuannya bukan membuat AI menjadi dokter, melainkan memastikan output tidak
melampaui cakupan sistem: tidak mendiagnosis, tidak meresepkan, tidak menutupi
kondisi gawat darurat, dan tidak menyatakan hal yang tidak didukung bukti.

Keputusan: PASS | MODIFY | BLOCK
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..contracts import EvidenceStatus, SafetyDecision, SafetyFlag
from ..lexicon import EMERGENCY_TERMS, HIGH_RISK_POPULATIONS

logger = logging.getLogger(__name__)


# --- pola risiko ------------------------------------------------------------

# Bahasa diagnosis pasti. Sistem hanya menyajikan asesmen awal.
_DIAGNOSIS_CERTAINTY = [
    r"\banda (menderita|mengidap|terkena|positif)\b",
    r"\bkamu (menderita|mengidap|terkena|positif)\b",
    r"\bini (adalah|merupakan) (penyakit|kondisi|kasus)\b",
    r"\b(pasti|dipastikan|sudah jelas|jelas sekali|tidak diragukan)\b.{0,30}\b(penyakit|kanker|diabetes|infeksi|tumor)\b",
    r"\bdiagnosis(nya)? (adalah|ialah)\b",
    r"\byou (have|are suffering from|are diagnosed with)\b",
    r"\bdiagnosis:\s*\w+",
]

# Instruksi berbahaya / di luar kewenangan sistem
_DANGEROUS_INSTRUCTIONS = [
    r"\b(hentikan|stop|berhenti minum|putus)\b.{0,25}\b(obat|terapi|pengobatan|insulin)\b",
    r"\b(gandakan|naikkan|tingkatkan|dobel)\b.{0,20}\bdosis\b",
    r"\btidak perlu ke dokter\b",
    r"\btidak usah (periksa|ke dokter|ke rumah sakit)\b",
    r"\bganti(lah)? obat\b.{0,20}\bdengan\b",
    r"\b(suntik|injeksi) sendiri\b",
    r"\bstop taking\b.{0,20}\b(medication|medicine)\b",
    r"\bdouble the dose\b",
]

# Rekomendasi terapi/resep spesifik (dosis obat), di luar cakupan
_PRESCRIPTIVE = [
    r"\b(minum|konsumsi|gunakan|pakai|take)\b[^.]{0,40}\b\d+\s?(mg|ml|mcg|gram|g)\b",
    r"\b\d+\s?(mg|ml|mcg)\b[^.]{0,30}\b(3|2|1|tiga|dua|satu)\s?(x|kali)\s?(sehari|per hari|sehari)\b",
    r"\bresep(kan)?\b.{0,20}\b(antibiotik|kortikosteroid|opioid)\b",
    r"\bsaya (resepkan|anjurkan) (obat|antibiotik)\b",
]

# Bahasa jaminan berlebihan
_OVERCONFIDENT = [
    r"\b(dijamin|pasti sembuh|100%|seratus persen|selalu berhasil|tidak akan pernah)\b",
    r"\b(guaranteed|always works|100 ?%)\b",
]

_EMERGENCY_ADVICE_ID = (
    "⚠️ PERINGATAN: Keluhan yang Anda sebutkan dapat menandakan kondisi "
    "gawat darurat. Segera hubungi layanan gawat darurat (119 di Indonesia) "
    "atau pergi ke UGD terdekat sekarang juga. Jangan menunda untuk mencari "
    "pertolongan medis."
)

_DIAGNOSIS_NOTICE_ID = (
    "Catatan: informasi di atas adalah asesmen awal berbasis literatur, "
    "bukan diagnosis. Penegakan diagnosis memerlukan pemeriksaan langsung "
    "oleh tenaga kesehatan."
)

_UNSUPPORTED_NOTICE_ID = (
    "Catatan: sebagian pernyataan di atas belum dapat ditelusuri langsung ke "
    "sumber yang tersedia di basis pengetahuan kami, sehingga perlu "
    "diperlakukan sebagai informasi umum, bukan kesimpulan berbasis bukti."
)

_BLOCK_MESSAGE_ID = (
    "Maaf, permintaan ini berada di luar cakupan layanan kami. Sistem ini "
    "tidak dapat memberikan instruksi pengobatan, dosis obat, atau tindakan "
    "medis. Silakan konsultasikan dengan dokter atau apoteker."
)


@dataclass
class SafetyReport:
    """Putusan lapisan keamanan beserta jawaban yang mungkin sudah disunting."""

    decision: SafetyDecision = SafetyDecision.PASS
    flags: List[SafetyFlag] = field(default_factory=list)
    answer: str = ""
    notes: List[str] = field(default_factory=list)

    def has_flag(self, code: str) -> bool:
        return any(f.code == code for f in self.flags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "flags": [f.to_dict() for f in self.flags],
            "notes": self.notes,
        }


def _matches(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            hits.append(match.group(0)[:80])
    return hits


def detect_emergency(text: str) -> List[str]:
    """
    Deteksi sinyal kegawatdaruratan pada keluhan pengguna.

    Dipakai pada teks yang ditulis pengguna (pesan atau transkrip percakapan),
    bukan pada jawaban yang dihasilkan sistem, lihat catatan di
    `validate_response`.
    """
    low = (text or "").lower()
    return [term for term in EMERGENCY_TERMS if term in low]


def detect_high_risk_population(text: str) -> List[str]:
    low = (text or "").lower()
    return [term for term in HIGH_RISK_POPULATIONS if term in low]


# Ditandai hanya bila tidak ada satu pun bagian jawaban yang tertelusur ke
# evidence. Menandai berdasarkan proporsi menghasilkan peringatan palsu, sebab
# jawaban Indonesia atas jurnal Inggris kerap gagal dicocokkan secara leksikal.


def validate_response(answer: str,
                      user_input: str = "",
                      evidence_status: Optional[EvidenceStatus] = None,
                      unsupported_claims: Optional[Iterable] = None,
                      intent: Optional[Any] = None,
                      total_claims: Optional[int] = None) -> SafetyReport:
    """
    Validasi jawaban sebelum dikirim ke consumer.

    Returns SafetyReport dengan `answer` yang sudah dimodifikasi bila perlu.
    """
    answer = answer or ""
    report = SafetyReport(decision=SafetyDecision.PASS, answer=answer)
    combined = f"{user_input}\n{answer}"

    # 1) Kegawatdaruratan, dideteksi hanya dari keluhan pengguna. Menyisir teks
    # jawaban memicu peringatan pada kalimat edukatif, dan peringatan yang
    # muncul di mana-mana justru diabaikan saat benar-benar dibutuhkan.
    emergency_hits = detect_emergency(user_input)
    if emergency_hits:
        report.flags.append(SafetyFlag(
            code="EMERGENCY_SIGNAL",
            severity="critical",
            message=f"Sinyal kegawatdaruratan terdeteksi: {', '.join(sorted(set(emergency_hits))[:5])}",
        ))
        if _EMERGENCY_ADVICE_ID not in report.answer:
            report.answer = f"{_EMERGENCY_ADVICE_ID}\n\n{report.answer}".strip()
        report.decision = SafetyDecision.MODIFY

    # 2) Instruksi berbahaya pada jawaban -> BLOCK.
    danger_hits = _matches(answer, _DANGEROUS_INSTRUCTIONS)
    if danger_hits:
        report.flags.append(SafetyFlag(
            code="DANGEROUS_INSTRUCTION",
            severity="critical",
            message=f"Instruksi berisiko terdeteksi: {'; '.join(danger_hits[:3])}",
        ))
        report.decision = SafetyDecision.BLOCK
        base = _BLOCK_MESSAGE_ID
        if emergency_hits:
            base = f"{_EMERGENCY_ADVICE_ID}\n\n{base}"
        report.answer = base
        return report

    # 3) Rekomendasi dosis/resep spesifik -> hapus bagian preskriptif.
    prescriptive_hits = _matches(answer, _PRESCRIPTIVE)
    if prescriptive_hits:
        report.flags.append(SafetyFlag(
            code="TREATMENT_RECOMMENDATION_RISK",
            severity="warning",
            message=f"Anjuran dosis/terapi spesifik dihapus: {'; '.join(prescriptive_hits[:3])}",
        ))
        report.answer = _strip_prescriptive_sentences(report.answer)
        report.answer = (
            report.answer.rstrip()
            + "\n\nUntuk pilihan dan dosis obat, konsultasikan dengan dokter atau apoteker."
        ).strip()
        report.decision = SafetyDecision.MODIFY

    # 4) Bahasa diagnosis pasti -> lunakkan + beri catatan.
    diagnosis_hits = _matches(answer, _DIAGNOSIS_CERTAINTY)
    if diagnosis_hits:
        report.flags.append(SafetyFlag(
            code="DIAGNOSIS_CERTAINTY",
            severity="warning",
            message=f"Bahasa diagnosis pasti dilunakkan: {'; '.join(diagnosis_hits[:3])}",
        ))
        report.answer = _soften_diagnosis(report.answer)
        if _DIAGNOSIS_NOTICE_ID not in report.answer:
            report.answer = f"{report.answer.rstrip()}\n\n{_DIAGNOSIS_NOTICE_ID}"
        report.decision = SafetyDecision.MODIFY

    # 5) Klaim berlebihan.
    overconfident_hits = _matches(answer, _OVERCONFIDENT)
    if overconfident_hits:
        report.flags.append(SafetyFlag(
            code="OVERCONFIDENT_CLAIM",
            severity="warning",
            message=f"Klaim berlebihan: {'; '.join(overconfident_hits[:3])}",
        ))
        report.decision = SafetyDecision.MODIFY

    # 6) Pernyataan tanpa dukungan bukti.
    unsupported = list(unsupported_claims or [])
    total = total_claims if total_claims is not None else len(unsupported)
    nothing_traceable = bool(total) and len(unsupported) >= total
    if (unsupported
            and evidence_status != EvidenceStatus.INSUFFICIENT_EVIDENCE
            and nothing_traceable):
        report.flags.append(SafetyFlag(
            code="UNSUPPORTED_MEDICAL_CLAIM",
            severity="warning",
            message=(
                f"Tidak ada satu pun dari {total} pernyataan yang dapat "
                "ditelusuri ke evidence."
            ),
        ))
        if _UNSUPPORTED_NOTICE_ID not in report.answer:
            report.answer = f"{report.answer.rstrip()}\n\n{_UNSUPPORTED_NOTICE_ID}"
        report.decision = SafetyDecision.MODIFY

    # 7) Bukti tidak cukup.
    if evidence_status == EvidenceStatus.INSUFFICIENT_EVIDENCE:
        report.flags.append(SafetyFlag(
            code="INSUFFICIENT_EVIDENCE",
            severity="info",
            message="Tidak ditemukan bukti yang cukup relevan di basis pengetahuan.",
        ))

    # 8) Kelompok rentan.
    risk_population = detect_high_risk_population(user_input)
    if risk_population:
        report.flags.append(SafetyFlag(
            code="HIGH_RISK_POPULATION",
            severity="warning",
            message=f"Kelompok rentan disebut: {', '.join(sorted(set(risk_population))[:4])}",
        ))

    # 9) Konteks obat.
    if re.search(r"\b(obat|dosis|antibiotik|medication|dose)\b", combined, re.IGNORECASE):
        report.flags.append(SafetyFlag(
            code="MEDICATION_CONTEXT",
            severity="info",
            message="Pertanyaan menyangkut obat, jawaban dibatasi pada informasi umum.",
        ))

    return report


def _strip_prescriptive_sentences(answer: str) -> str:
    """Buang kalimat yang memuat anjuran dosis spesifik."""
    sentences = re.split(r"(?<=[.!?])\s+", answer or "")
    kept = [
        s for s in sentences
        if not any(re.search(p, s, re.IGNORECASE) for p in _PRESCRIPTIVE)
    ]
    return " ".join(kept).strip() or answer


def _soften_diagnosis(answer: str) -> str:
    """Ganti frasa diagnosis pasti dengan bahasa kemungkinan."""
    replacements = [
        (r"\banda (menderita|mengidap|terkena|positif)\b", "keluhan Anda dapat sejalan dengan"),
        (r"\bkamu (menderita|mengidap|terkena|positif)\b", "keluhan Anda dapat sejalan dengan"),
        (r"\bdiagnosis(nya)? (adalah|ialah)\b", "kemungkinan yang perlu dievaluasi adalah"),
        (r"\bini (adalah|merupakan) (penyakit|kondisi|kasus)\b", r"ini dapat berkaitan dengan \2"),
        (r"\byou (have|are suffering from|are diagnosed with)\b", "your symptoms may be consistent with"),
        (r"\b(pasti|dipastikan|sudah jelas|tidak diragukan)\b", "kemungkinan"),
    ]
    result = answer or ""
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result
