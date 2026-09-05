"""
Response engine.

Generation DIPISAH dari retrieval. Urutan yang dijamin modul ini:

    Understanding -> Context -> Evidence Retrieval -> Evidence Validation
                  -> Response Generation

LLM hanya boleh menyusun kalimat dari evidence yang sudah divalidasi.
Bila evidence tidak cukup, LLM tidak dipanggil sama sekali dan sistem
mengembalikan jawaban transparan bahwa bukti tidak memadai.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import (
    PRELIMINARY_ASSESSMENT_DISCLAIMER,
    PRELIMINARY_ASSESSMENT_STATUS,
    EvidenceItem,
    EvidenceStatus,
    HealthContext,
    Intent,
    Mode,
)
from . import llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Anda adalah asisten informasi kesehatan berbasis bukti untuk platform Healthify. "
    "Anda BUKAN dokter dan tidak boleh memberikan diagnosis, resep, atau dosis obat.\n"
    "Aturan mutlak:\n"
    "1. Gunakan hanya informasi dari daftar EVIDENCE yang diberikan.\n"
    "2. DILARANG menyebut, mengarang, atau menambahkan sumber, DOI, URL, nama jurnal, "
    "penulis, atau tahun yang tidak ada di daftar EVIDENCE.\n"
    "3. Rujuk evidence dengan penanda [E1], [E2], dst tepat setelah kalimat yang didukungnya.\n"
    "4. Jika EVIDENCE tidak menjawab pertanyaan, katakan terus terang bahwa bukti yang "
    "tersedia belum memadai. Jangan menebak.\n"
    "5. Jawab dalam bahasa yang sama dengan pertanyaan pengguna (default: Bahasa Indonesia).\n"
    "6. Nyatakan ketidakpastian secara eksplisit bila bukti terbatas atau bertentangan.\n"
    "7. Tulis untuk PEMBACA AWAM, bukan untuk sesama sistem. Jangan menarasikan "
    "proses: dilarang menulis \"EVIDENCE\", \"bukti yang tersedia\", \"berdasarkan "
    "evidence di atas\", \"dari sumber yang diberikan\", atau menyebut nomor sumber "
    "di luar penanda [E1]. Langsung sampaikan informasinya.\n"
    "8. Jangan menawarkan bantuan lanjutan atau bertanya balik kecuali memang perlu "
    "satu klarifikasi penting.\n"
    "9. Penanda [E1] adalah CATATAN KAKI, bukan kata dalam kalimat. Letakkan di "
    "akhir kalimat dan jangan pernah menjadikannya subjek.\n"
    "   BENAR : \"Demam berdarah disebabkan oleh virus dengue yang ditularkan nyamuk. [E1]\"\n"
    "   SALAH : \"[E1] menyatakan bahwa demam berdarah disebabkan virus dengue.\"\n"
    "10. Jangan mengomentari cakupan tiap sumber satu per satu. Bila sesuatu tidak "
    "diketahui, cukup satu kalimat singkat, lalu sampaikan yang memang diketahui."
)

_INSUFFICIENT_TEMPLATE_ID = (
    "Saat ini kami belum menemukan bukti ilmiah yang cukup relevan di basis "
    "pengetahuan Healthify untuk menjawab pertanyaan tersebut secara "
    "bertanggung jawab.\n\n"
    "Yang dapat kami sampaikan:\n"
    "- Kami tidak menebak jawaban ketika bukti pendukung tidak tersedia.\n"
    "- Informasi ini bukan berarti pertanyaan Anda salah atau tidak penting.\n\n"
    "Langkah yang disarankan: konsultasikan keluhan atau pertanyaan ini dengan "
    "tenaga kesehatan agar dapat dinilai secara langsung."
)

_UNSUPPORTED_TEMPLATE_ID = (
    "Maaf, pertanyaan ini berada di luar cakupan layanan informasi kesehatan "
    "Healthify. Sistem ini hanya menangani pertanyaan dan klaim seputar "
    "kesehatan berbasis literatur ilmiah."
)

_SMALL_TALK_TEMPLATE_ID = (
    "Sama-sama. Silakan sampaikan bila ada yang ingin ditanyakan seputar "
    "kesehatan, dan saya akan mencarikan rujukannya dari literatur ilmiah."
)

_GREETING_TEMPLATE_ID = (
    "Halo. Silakan sampaikan pertanyaan atau keluhan kesehatan Anda, dan saya "
    "akan mencarikan jawabannya beserta rujukan jurnalnya."
)

_CITATION_RE = re.compile(r"\[E(\d{1,2})\]")


# Penyusunan prompt

def format_evidence_block(evidence: List[EvidenceItem], max_chars: int = 900) -> str:
    """Susun daftar evidence bernomor untuk prompt."""
    lines: List[str] = []
    for idx, item in enumerate(evidence, start=1):
        title = item.title or "(tanpa judul)"
        snippet = (item.snippet or "").replace("\n", " ").strip()[:max_chars]
        meta = []
        if item.published_year:
            meta.append(str(item.published_year))
        if item.publisher:
            meta.append(item.publisher[:60])
        meta_str = f" ({', '.join(meta)})" if meta else ""
        lines.append(f"[E{idx}] {title}{meta_str}\n{snippet}")
    return "\n\n".join(lines)


def format_health_context(context: HealthContext) -> str:
    """Ringkas health context untuk prompt. Hanya field yang terisi."""
    if context.is_empty():
        return ""
    fields = [
        ("Keluhan utama", context.chief_complaint),
        ("Gejala", ", ".join(context.symptoms) if context.symptoms else None),
        ("Durasi", context.duration),
        ("Tingkat keparahan", context.severity),
        ("Awitan", context.onset),
        ("Perkembangan", context.progression),
        ("Obat yang disebut", ", ".join(context.medications) if context.medications else None),
        ("Alergi", ", ".join(context.allergies) if context.allergies else None),
        ("Riwayat relevan", ", ".join(context.relevant_history) if context.relevant_history else None),
    ]
    lines = [f"- {label}: {value}" for label, value in fields if value]
    return "\n".join(lines)


def format_history(previous_messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    if not previous_messages:
        return ""
    recent = previous_messages[-max_turns:]
    return "\n".join(
        f"{'Pengguna' if m.get('role') == 'user' else 'Asisten'}: {m.get('content', '')[:300]}"
        for m in recent
    )


_INTENT_INSTRUCTION = {
    Intent.CLAIM_VERIFICATION: (
        "Tugas: nilai klaim pengguna terhadap EVIDENCE. Nyatakan apakah klaim "
        "DIDUKUNG, tidak DIDUKUNG, atau BUKTI tidak KONKLUSIF, lalu jelaskan alasannya."
    ),
    Intent.SYMPTOM_CONTEXT: (
        "Tugas: tanggapi keluhan pengguna dengan informasi umum berbasis EVIDENCE. "
        "Jangan mendiagnosis. Boleh menyebutkan hal-hal yang perlu diperhatikan dan "
        "kapan sebaiknya memeriksakan diri. Jika informasi penting belum diketahui "
        "(durasi, keparahan, gejala penyerta), tanyakan maksimal 2 pertanyaan klarifikasi."
    ),
    Intent.HEALTH_INFORMATION: (
        "Tugas: jawab pertanyaan informasi kesehatan secara ringkas dan akurat "
        "berdasarkan EVIDENCE."
    ),
    Intent.FOLLOW_UP: (
        "Tugas: jawab pertanyaan lanjutan ini dengan tetap merujuk pada konteks "
        "percakapan sebelumnya. Jangan mengulang seluruh penjelasan sebelumnya."
    ),
    Intent.MEDICATION_INFORMATION: (
        "Tugas: berikan informasi umum tentang obat yang ditanyakan berdasarkan EVIDENCE "
        "(kegunaan umum, hal yang perlu diperhatikan). DILARANG memberikan dosis, "
        "aturan pakai, atau anjuran mengganti/menghentikan obat."
    ),
    Intent.GENERAL_HEALTH: (
        "Tugas: berikan informasi kesehatan umum yang relevan berdasarkan EVIDENCE."
    ),
}


def build_prompt(query: str, intent: Intent, context: HealthContext,
                 evidence: List[EvidenceItem],
                 previous_messages: Optional[List[Dict[str, str]]] = None,
                 language: str = "id") -> str:
    sections: List[str] = []

    instruction = _INTENT_INSTRUCTION.get(intent, _INTENT_INSTRUCTION[Intent.GENERAL_HEALTH])
    sections.append(instruction)

    history = format_history(previous_messages or [])
    if history:
        sections.append(f"RIWAYAT PERCAKAPAN:\n{history}")

    ctx = format_health_context(context)
    if ctx:
        sections.append(f"KONTEKS KESEHATAN YANG DILAPORKAN PENGGUNA:\n{ctx}")

    sections.append(f"PERTANYAAN PENGGUNA:\n{query}")
    sections.append(f"EVIDENCE:\n{format_evidence_block(evidence)}")

    sections.append(
        "FORMAT JAWABAN:\n"
        "- 3-6 kalimat, bahasa jelas untuk pembaca awam dan tidak menakut-nakuti.\n"
        "- Mulai langsung dengan informasinya, bukan dengan komentar tentang bukti.\n"
        "- Sertakan penanda [E1]/[E2] di AKHIR kalimat yang bersumber dari evidence, "
        "tidak pernah di awal kalimat.\n"
        "- Bila informasi yang ditanyakan tidak ada, katakan singkat bahwa hal itu "
        "belum dapat dipastikan, lalu sampaikan yang memang diketahui.\n"
        "- Jangan menuliskan daftar pustaka; sumber ditampilkan terpisah oleh sistem."
    )
    if language == "en":
        sections.append("Jawab dalam Bahasa Inggris.")

    return "\n\n".join(sections)


# Fallback deterministik (tanpa LLM)

def _first_sentences(text: str, count: int = 2, max_chars: int = 320) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    return " ".join(sentences[:count])[:max_chars]


def build_extractive_answer(query: str, intent: Intent, context: HealthContext,
                            evidence: List[EvidenceItem]) -> str:
    """
    Susun jawaban tanpa LLM, murni dari cuplikan evidence.

    Dipakai bila tidak ada provider LLM yang dikonfigurasi. Sengaja konservatif:
    hanya merangkai kutipan, tanpa menyimpulkan hal yang tidak tertulis.
    """
    if not evidence:
        return _INSUFFICIENT_TEMPLATE_ID

    opening = {
        Intent.CLAIM_VERIFICATION: "Berikut bukti yang ditemukan terkait klaim tersebut:",
        Intent.SYMPTOM_CONTEXT: "Berikut informasi dari literatur yang relevan dengan keluhan Anda:",
        Intent.MEDICATION_INFORMATION: "Berikut informasi umum dari literatur mengenai obat yang Anda tanyakan:",
        Intent.FOLLOW_UP: "Melanjutkan pembahasan sebelumnya, berikut yang dapat kami sampaikan:",
    }.get(intent, "Berikut informasi dari literatur yang relevan dengan pertanyaan Anda:")

    parts = [opening]
    for idx, item in enumerate(evidence[:3], start=1):
        excerpt = _first_sentences(item.snippet)
        if not excerpt:
            continue
        title = item.title or "sumber terkait"
        parts.append(f"- {excerpt} [E{idx}] (dari: {title[:120]})")

    parts.append(
        "Ringkasan ini disusun langsung dari cuplikan sumber di atas tanpa "
        "penafsiran tambahan. Untuk penilaian yang sesuai kondisi Anda, "
        "konsultasikan dengan tenaga kesehatan."
    )
    return "\n".join(parts)


# Generation utama

def generate_response(query: str,
                      intent: Intent,
                      context: HealthContext,
                      evidence: List[EvidenceItem],
                      evidence_status: EvidenceStatus,
                      previous_messages: Optional[List[Dict[str, str]]] = None,
                      language: str = "id") -> Tuple[str, Dict[str, Any]]:
    """
    Hasilkan jawaban final.

    Returns: (answer, metadata)
    """
    meta: Dict[str, Any] = {
        "generator": "none",
        "llm_provider": llm.available_provider(),
        "evidence_used": len(evidence),
    }

    if intent == Intent.UNSUPPORTED:
        meta["generator"] = "template_unsupported"
        return _UNSUPPORTED_TEMPLATE_ID, meta

    # Basa-basi dijawab seperlunya, tanpa pencarian literatur.
    if intent == Intent.SMALL_TALK:
        meta["generator"] = "template_small_talk"
        greeting = bool(re.match(r"^\W*(halo|hai|hi|hello|selamat\s+(pagi|siang|sore|malam)|assalam)",
                                 (query or "").strip().lower()))
        return (_GREETING_TEMPLATE_ID if greeting else _SMALL_TALK_TEMPLATE_ID), meta

    #, bukti tidak cukup, jadi LLM tidak diminta menebak.
    if evidence_status == EvidenceStatus.INSUFFICIENT_EVIDENCE or not evidence:
        meta["generator"] = "template_insufficient_evidence"
        return _INSUFFICIENT_TEMPLATE_ID, meta

    prompt = build_prompt(query, intent, context, evidence, previous_messages, language)
    answer = llm.generate(prompt, system=_SYSTEM_PROMPT, temperature=0.15, max_tokens=900)

    if answer:
        meta["generator"] = f"llm:{meta['llm_provider']}"
        answer = _strip_fabricated_references(answer, evidence)
        answer = _normalize_citation_placement(answer)
    else:
        meta["generator"] = "extractive_fallback"
        answer = build_extractive_answer(query, intent, context, evidence)

    if evidence_status == EvidenceStatus.PARTIAL:
        answer = (
            f"{answer.rstrip()}\n\nCatatan: bukti yang tersedia terbatas, "
            "sehingga kesimpulan di atas belum dapat dianggap final."
        )

    return answer.strip(), meta


# Model kadang memakai penanda sitasi sebagai SUBJEK kalimat
# ("[E1] menyatakan bahwa demam berdarah disebabkan virus dengue").
# Penanda seperti itu tidak bisa sekadar dihapus, kalimatnya akan kehilangan
# subjek ("secara langsung menyatakan bahwa ..."). Normalisasi di bawah
# mengubahnya menjadi kalimat utuh dengan penanda pindah ke akhir.
_TRAILING_CITATION_RE = re.compile(r"\[E\d{1,2}\]\s*$")

# Kata kerja pelaporan yang menjadikan sumber sebagai pelaku kalimat.
_REPORTING_VERBS = (
    r"(?:secara langsung\s+|juga\s+|hanya\s+|sama-sama\s+|memang\s+)*"
    r"(?:menyatakan|menyebutkan|menyebut|menjelaskan|melaporkan|menunjukkan|"
    r"membahas|mengatakan|mencatat|menemukan|mengindikasikan)\s+(?:bahwa\s+)?"
)

# Awal kalimat yang menjadikan sumber sebagai SUBJEK. Tiga bentuk ditemui:
#   "[E1] menyatakan bahwa ..."   penanda berkurung
#   "E1 hanya menyebut bahwa ..." penanda tanpa kurung
#   "Evidence menyatakan bahwa ..." kata benda internal
# Ketiganya diubah menjadi kalimat biasa, penanda dipindah ke akhir.
_LEADING_CITATION_RE = re.compile(
    r"^\s*(?:"
    r"(?P<markers>(?:\[E\d{1,2}\]|\bE\d{1,2}\b)(?:\s*(?:dan|,|&)?\s*(?:\[E\d{1,2}\]|\bE\d{1,2}\b))*)"
    r"|(?P<subject>(?:evidence|bukti|sumber|referensi|studi tersebut|literatur)"
    r"(?:\s+(?:yang\s+)?(?:tersedia|di atas|tersebut|ini|yang diberikan))?)"
    r")\s*(?P<reporting>" + _REPORTING_VERBS + r")?",
    re.IGNORECASE,
)
_MARKER_NUMBER_RE = re.compile(r"E(\d{1,2})", re.IGNORECASE)


def _normalize_citation_placement(answer: str) -> str:
    """
    Pastikan rujukan sumber berperan sebagai catatan kaki, bukan subjek kalimat.

    Model kadang menulis "[E1] menyatakan bahwa X", "E1 menyebut bahwa X", atau
    "Evidence menunjukkan bahwa X". Ketiganya membocorkan mekanika internal ke
    pembaca, dan yang pertama menjadi penggalan tak bersubjek begitu penanda
    dibersihkan untuk ditampilkan. Semua diubah menjadi "X. [E1]".
    """
    from ..evidence.provenance import split_sentences

    if not answer:
        return answer

    rebuilt = []
    changed = False
    for sentence in split_sentences(answer):
        match = _LEADING_CITATION_RE.match(sentence)
        if not match or not (match.group("markers") or match.group("subject")):
            rebuilt.append(sentence)
            continue

        # Bila yang cocok hanya kata, bukan penanda, diperlukan kata kerja
        # pelaporan. Tanpa syarat ini kalimat sah seperti "Bukti ini penting
        # dipahami." ikut dipotong dan kehilangan makna.
        if match.group("subject") and not match.group("reporting"):
            rebuilt.append(sentence)
            continue

        body = sentence[match.end():].strip()
        if not body:
            rebuilt.append(sentence)
            continue

        changed = True
        body = body[0].upper() + body[1:]

        raw_markers = match.group("markers") or ""
        numbers = _MARKER_NUMBER_RE.findall(raw_markers)
        markers = "".join(f"[E{n}]" for n in numbers)

        if markers and not _TRAILING_CITATION_RE.search(body):
            rebuilt.append(f"{body} {markers}")
        else:
            rebuilt.append(body)

    if not changed:
        return answer
    return " ".join(rebuilt) if rebuilt else answer


_URL_IN_TEXT_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DOI_IN_TEXT_RE = re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+", re.IGNORECASE)


def _strip_fabricated_references(answer: str, evidence: List[EvidenceItem]) -> str:
    """
    Buang URL/DOI apa pun yang ditulis LLM di badan jawaban.

    Sumber resmi hanya boleh datang dari field `evidence` pada response, yang
    seluruhnya berasal dari knowledge base dan sudah tervalidasi. Ini menutup
    celah terakhir munculnya DOI karangan di teks jawaban.
    """
    allowed_dois = {(e.doi or "").lower() for e in evidence if e.doi}
    allowed_urls = {(e.url or "").lower() for e in evidence if e.url}

    def replace_url(match):
        return match.group(0) if match.group(0).lower() in allowed_urls else ""

    def replace_doi(match):
        raw = match.group(0)
        normalized = re.sub(r"^doi:\s*", "", raw, flags=re.IGNORECASE).lower()
        return raw if normalized in allowed_dois else ""

    cleaned = _URL_IN_TEXT_RE.sub(replace_url, answer or "")
    cleaned = _DOI_IN_TEXT_RE.sub(replace_doi, cleaned)
    # rapikan sisa tanda baca/spasi
    cleaned = re.sub(r"\(\s*[,;]?\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;])", r"\1", cleaned)
    return cleaned.strip()


def citation_map(answer: str, evidence: List[EvidenceItem]) -> Dict[int, EvidenceItem]:
    """Petakan penanda [En] pada jawaban ke EvidenceItem-nya."""
    mapping: Dict[int, EvidenceItem] = {}
    for match in _CITATION_RE.finditer(answer or ""):
        idx = int(match.group(1))
        if 1 <= idx <= len(evidence):
            mapping[idx] = evidence[idx - 1]
    return mapping


# Preliminary assessment

def build_preliminary_assessment(context: HealthContext,
                                 evidence: List[EvidenceItem],
                                 evidence_status: EvidenceStatus,
                                 safety_flags: Optional[List] = None) -> Optional[Dict[str, Any]]:
    """
    Susun asesmen awal, BUKAN diagnosis.

    Hanya dibuat bila ada konteks gejala nyata dari user. Isinya sengaja
    tidak menyebut nama penyakit sebagai kesimpulan; ia merangkum apa yang
    dilaporkan, seberapa kuat dukungan bukti, dan langkah yang disarankan.
    """
    if context.is_empty() or not context.symptoms:
        return None

    urgency = "routine"
    flags = safety_flags or []
    if any(getattr(f, "code", "") == "EMERGENCY_SIGNAL" for f in flags):
        urgency = "emergency"
    elif any(getattr(f, "code", "") == "HIGH_RISK_POPULATION" for f in flags):
        urgency = "elevated"
    elif context.severity and "berat" in (context.severity or "").lower():
        urgency = "elevated"
    elif context.progression == "memburuk":
        urgency = "elevated"

    confidence = {
        EvidenceStatus.SUFFICIENT: "moderate",
        EvidenceStatus.PARTIAL: "low",
        EvidenceStatus.INSUFFICIENT_EVIDENCE: "very_low",
    }[evidence_status]

    next_steps: List[str] = []
    if urgency == "emergency":
        next_steps.append("Segera cari pertolongan gawat darurat (119/UGD terdekat).")
    else:
        if not context.duration:
            next_steps.append("Catat sejak kapan keluhan muncul.")
        if not context.severity:
            next_steps.append("Perhatikan tingkat keparahan dan dampaknya pada aktivitas harian.")
        next_steps.append(
            "Periksakan diri ke fasilitas kesehatan bila keluhan menetap, memberat, "
            "atau muncul gejala baru."
        )

    return {
        "status": PRELIMINARY_ASSESSMENT_STATUS,
        "is_diagnosis": False,
        "disclaimer": PRELIMINARY_ASSESSMENT_DISCLAIMER,
        "reported_symptoms": list(context.symptoms),
        "duration": context.duration,
        "severity": context.severity,
        "progression": context.progression,
        "urgency": urgency,
        "evidence_confidence": confidence,
        "evidence_count": len(evidence),
        "recommended_next_step": next_steps,
    }
