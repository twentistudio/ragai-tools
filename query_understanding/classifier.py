"""
Query understanding (§7).

Menentukan *kebutuhan informasi* user sehingga pipeline yang tepat bisa dipilih.
Klasifikasi ini deterministik (rule + leksikon) supaya:
  - hasilnya stabil dan bisa diuji,
  - tidak menambah latency/biaya LLM di jalur panas,
  - tidak pernah "mengarang" intent.

Intent TIDAK mengubah kebenaran jawaban; ia hanya memilih pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..contracts import Intent, Mode
from ..lexicon import (
    ALL_SYMPTOM_VARIANTS,
    CONDITION_TERMS,
    GENERAL_HEALTH_TERMS,
    MEDICATION_TERMS,
    OUT_OF_SCOPE_TERMS,
    SYMPTOM_LOOKUP,
    find_symptom_variants,
)

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Hasil klasifikasi intent beserta keyakinan dan sinyal pendukungnya."""

    intent: Intent
    confidence: float = 0.5
    signals: List[str] = field(default_factory=list)
    is_health_related: bool = True

    def to_dict(self) -> Dict:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
            "is_health_related": self.is_health_related,
        }


# --- pola-pola sinyal -------------------------------------------------------

# Pertanyaan lanjutan yang bergantung pada konteks sebelumnya
_FOLLOWUP_PATTERNS = [
    r"\b(itu|tersebut|tadi|sebelumnya|barusan)\b",
    r"\bdari (penjelasan|jawaban|informasi) (di atas|sebelumnya|tadi)\b",
    r"\bkalau (begitu|gitu|dari)\b",
    r"^\s*(lalu|terus|kemudian|selanjutnya|jadi)\b",
    r"\bapakah (itu|ini) (normal|bahaya|berbahaya|wajar)\b",
    r"\bbagaimana dengan\b",
    r"\b(that|it|this|above|previous|earlier)\b.*\?$",
    r"^\s*(what about|and then|so)\b",
]

# Sinyal KUAT bahwa pengguna meminta verifikasi: ada klaim beredar yang
# ingin dinilai benar/salah.
_CLAIM_STRONG_PATTERNS = [
    r"\b(benarkah|apakah benar|betulkah|apa betul)\b",
    r"\b(hoax|hoaks|mitos|myth|fakta atau|palsu|bohong)\b",
    r"\b(katanya|konon|banyak yang bilang|beredar (kabar|informasi)|viral)\b",
    r"\b(is it true|true or false|debunk)\b",
]

# Pola PERNYATAAN sebab-akibat. Ini hanya klaim bila ditulis sebagai
# pernyataan — bukan ketika dipakai dalam pertanyaan informasi.
#
# "Vitamin C menyembuhkan kanker."          -> klaim, perlu diverifikasi
# "Apa yang menyebabkan demam berdarah?"    -> pertanyaan informasi biasa
# "Bagaimana cara mencegah ISK?"            -> pertanyaan informasi biasa
_CLAIM_ASSERTION_PATTERNS = [
    r"\b\w+\s+(dapat|bisa|mampu)\s+(menyembuhkan|mengobati|mencegah|membunuh)\b",
    r"\b\w+\s+(menyembuhkan|mengobati|menyebabkan|memicu|mencegah)\s+\w+",
    r"\b(cures?|causes?|prevents?)\b",
]

# Pertanyaan yang menanyakan informasi, bukan meminta penilaian klaim.
_INFORMATION_QUESTION_RE = re.compile(
    r"^\s*(apa|apakah saja|apa saja|bagaimana|gimana|kenapa|mengapa|kapan|"
    r"di ?mana|siapa|berapa|sebutkan|jelaskan|"
    r"what|how|why|when|where|who|which)\b",
    re.IGNORECASE,
)

# Laporan gejala orang pertama
_SYMPTOM_SELF_PATTERNS = [
    r"\b(saya|aku|gue|gua|kami|anak saya|ibu saya|ayah saya|istri saya|suami saya)\b",
    r"\b(i|my|i'm|im)\b\s+(have|has|am|feel|got)",
    r"\b(mengalami|merasakan|menderita|terasa)\b",
    r"\bsudah\s+\d+\s+(hari|minggu|bulan|jam)\b",
    r"\bsejak\s+\d*\s*(hari|minggu|bulan|kemarin|tadi)\b",
]

# Permintaan informasi kesehatan umum
_INFORMATION_PATTERNS = [
    r"\b(apa itu|apakah yang dimaksud|apa penyebab|apa gejala|apa saja)\b",
    r"\b(bagaimana cara|gimana cara|bagaimana|cara)\b",
    r"\b(kapan (harus|perlu|sebaiknya))\b",
    r"\b(perlukah|apakah perlu|haruskah|apakah harus)\b",
    r"\b(what is|how (to|do|does)|when should|why does)\b",
    r"\b(butuh|membutuhkan|memerlukan)\s+(pemeriksaan|tes|periksa|cek)\b",
]

_MEDICATION_PATTERNS = [
    r"\b(dosis|dose|dosage|takaran)\b",
    r"\b(efek samping|side effects?)\b",
    r"\b(interaksi obat|drug interaction)\b",
    r"\b(boleh|aman)\s+(minum|konsumsi|pakai)\b",
    r"\b(berapa kali|berapa banyak)\s+(sehari|per hari|minum)\b",
]

_QUESTION_RE = re.compile(r"\?|^\s*(apa|apakah|bagaimana|gimana|kenapa|mengapa|kapan|di ?mana|siapa|berapa|bisakah|apakah|what|how|why|when|where|who|can|should|is|are|does)\b", re.IGNORECASE)


def _count_matches(text: str, patterns: List[str]) -> List[str]:
    hits = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            hits.append(pattern)
    return hits


def _contains_any(text: str, terms) -> List[str]:
    return [t for t in terms if t in text]


def is_health_related(text: str) -> bool:
    """Deteksi apakah teks berkaitan dengan kesehatan."""
    low = (text or "").lower()
    if not low.strip():
        return False
    if find_symptom_variants(low):
        return True
    if _contains_any(low, CONDITION_TERMS):
        return True
    if _contains_any(low, MEDICATION_TERMS):
        return True
    if _contains_any(low, GENERAL_HEALTH_TERMS):
        return True
    return False


def is_out_of_scope(text: str) -> bool:
    low = (text or "").lower()
    out_hits = _contains_any(low, OUT_OF_SCOPE_TERMS)
    return bool(out_hits) and not is_health_related(low)


# Basa-basi percakapan: sapaan, ucapan terima kasih, dan penutup. Dikenali
# hanya bila pesannya memang hanya itu — "halo dok, saya batuk seminggu" tetap
# laporan gejala, bukan sapaan.
_SMALL_TALK_PATTERNS = [
    re.compile(r"^\W*(terima\s*kasih|makasih|makasi|thanks|thank\s*you|thx|tengkyu)\b"),
    re.compile(r"^\W*(halo|hai|hi|hello|hey)\b"),
    re.compile(r"^\W*selamat\s+(pagi|siang|sore|malam)\b"),
    re.compile(r"^\W*(assalam|salam)\b"),
    re.compile(r"^\W*(oke?|ok|okay|baik|siap|sip|mantap|noted)\W*$"),
    re.compile(r"^\W*(sampai\s+jumpa|bye|dadah|permisi\s*$)"),
]

# Panjang maksimum sebuah pesan agar masih dianggap basa-basi. Kalimat yang
# lebih panjang hampir selalu membawa isi, sekalipun diawali sapaan.
_SMALL_TALK_MAX_WORDS = 6


def _is_small_talk(low: str, symptom_hits) -> bool:
    """
    Apakah pesan ini hanya basa-basi.

    Polanya cocok, pesannya pendek, dan tidak menyebut gejala maupun konsep
    kesehatan. Tanpa dua syarat terakhir, "halo dok, saya batuk seminggu"
    dijawab sebagai sapaan dan keluhannya hilang.
    """
    if symptom_hits:
        return False
    if len(low.split()) > _SMALL_TALK_MAX_WORDS:
        return False
    if not any(pattern.search(low) for pattern in _SMALL_TALK_PATTERNS):
        return False

    from ..retrieval.concepts import extract_health_concepts

    return not extract_health_concepts(low)


def classify_intent(query: str,
                    mode: Optional[Mode] = None,
                    previous_messages: Optional[List[Dict[str, str]]] = None) -> IntentResult:
    """
    Tentukan intent dari query (+ konteks percakapan bila ada).

    Urutan keputusan:
      1. Query kosong / di luar cakupan  -> UNSUPPORTED
      2. Referensi ke percakapan sebelumnya -> FOLLOW_UP
      3. Pola klaim                      -> CLAIM_VERIFICATION
      4. Sinyal obat kuat                -> MEDICATION_INFORMATION
      5. Laporan gejala orang pertama    -> SYMPTOM_CONTEXT
      6. Mode claim eksplisit            -> CLAIM_VERIFICATION
      7. Pola pertanyaan informasi       -> HEALTH_INFORMATION
      8. Sisanya (masih topik kesehatan) -> GENERAL_HEALTH
    """
    text = (query or "").strip()
    low = text.lower()
    previous_messages = previous_messages or []
    signals: List[str] = []

    if not text:
        return IntentResult(Intent.UNSUPPORTED, 0.99, ["empty_query"], False)

    health = is_health_related(low)

    if is_out_of_scope(low):
        return IntentResult(Intent.UNSUPPORTED, 0.9, ["out_of_scope_topic"], False)

    # Didahulukan atas seluruh pola lain: tanpa ini "terima kasih banyak"
    # menempuh pencarian literatur lengkap dan kembali membawa lima jurnal.
    if _is_small_talk(low, find_symptom_variants(low)):
        return IntentResult(Intent.SMALL_TALK, 0.9, ["conversational_courtesy"], True)

    # Mode eksplisit dari caller adalah sinyal kuat, tapi bukan penentu mutlak.
    mode = Mode.coerce(mode) if mode is not None else None

    # 2) FOLLOW_UP — hanya masuk akal bila ada riwayat percakapan.
    followup_hits = _count_matches(low, _FOLLOWUP_PATTERNS)
    symptom_hits = find_symptom_variants(low)
    if previous_messages and followup_hits:
        # "Saya demam" + "sudah tiga hari" -> tetap SYMPTOM_CONTEXT bila
        # kalimatnya membawa informasi gejala baru.
        if not symptom_hits or len(low.split()) <= 8:
            signals.append("followup_reference")
            return IntentResult(Intent.FOLLOW_UP, 0.8, signals, health or True)

    # Kalimat sangat pendek tanpa subjek setelah ada percakapan -> follow up
    if previous_messages and len(low.split()) <= 5 and not symptom_hits and _QUESTION_RE.search(low):
        return IntentResult(Intent.FOLLOW_UP, 0.65, ["short_contextual_question"], True)

    strong_claim_hits = _count_matches(low, _CLAIM_STRONG_PATTERNS)
    assertion_hits = _count_matches(low, _CLAIM_ASSERTION_PATTERNS)
    is_information_question = bool(_INFORMATION_QUESTION_RE.match(low))

    # Pola sebab-akibat di dalam pertanyaan informasi ("Apa yang menyebabkan X?")
    # bukan klaim yang perlu diverifikasi. Caller yang menyatakan
    # mode=information juga tidak boleh dipaksa masuk jalur verifikasi.
    treat_assertion_as_claim = bool(assertion_hits) and not (
        is_information_question or mode in (Mode.INFORMATION, Mode.CONSULTATION)
    )
    claim_hits = strong_claim_hits or (assertion_hits if treat_assertion_as_claim else [])
    self_hits = _count_matches(low, _SYMPTOM_SELF_PATTERNS)

    # 3) CLAIM_VERIFICATION — didahulukan.
    #    Klaim tentang obat/vitamin ("Vitamin C dosis tinggi menyembuhkan kanker")
    #    tetap sebuah klaim yang perlu diverifikasi, bukan permintaan informasi obat.
    #    Pengecualian: laporan gejala orang pertama tetap SYMPTOM_CONTEXT.
    if claim_hits and not (symptom_hits and self_hits):
        signals.append("claim_strong" if strong_claim_hits else "claim_assertion")
        return IntentResult(Intent.CLAIM_VERIFICATION, 0.9, signals, health)

    # 4) MEDICATION_INFORMATION
    med_terms = _contains_any(low, MEDICATION_TERMS)
    med_pattern_hits = _count_matches(low, _MEDICATION_PATTERNS)
    if med_terms and (med_pattern_hits or mode == Mode.MEDICATION or len(med_terms) >= 2):
        signals.extend(["medication_term"] * min(len(med_terms), 3))
        if med_pattern_hits:
            signals.append("medication_question")
        return IntentResult(Intent.MEDICATION_INFORMATION, 0.85, signals, True)
    if mode == Mode.MEDICATION and med_terms:
        return IntentResult(Intent.MEDICATION_INFORMATION, 0.7, ["mode_medication"], True)

    # 5) SYMPTOM_CONTEXT — laporan gejala orang pertama.
    if symptom_hits and self_hits and not claim_hits:
        signals.append("first_person_symptom")
        signals.extend(f"symptom:{SYMPTOM_LOOKUP.get(s, s)}" for s in symptom_hits[:4])
        return IntentResult(Intent.SYMPTOM_CONTEXT, 0.88, signals, True)

    # Durasi tanpa subjek eksplisit ("sudah tiga hari") saat mode consultation
    if symptom_hits and mode == Mode.CONSULTATION and not claim_hits and not _QUESTION_RE.search(low):
        signals.append("symptom_statement")
        return IntentResult(Intent.SYMPTOM_CONTEXT, 0.7, signals, True)

    # 6) Mode claim eksplisit dari caller.
    if mode == Mode.CLAIM:
        signals.append("mode_claim")
        return IntentResult(Intent.CLAIM_VERIFICATION, 0.7, signals, health)

    # 7) HEALTH_INFORMATION
    info_hits = _count_matches(low, _INFORMATION_PATTERNS)
    if health and (info_hits or _QUESTION_RE.search(low)):
        signals.append("information_question")
        return IntentResult(Intent.HEALTH_INFORMATION, 0.8, signals, True)

    # 8) sisanya
    if health:
        return IntentResult(Intent.GENERAL_HEALTH, 0.6, ["health_topic"], True)

    return IntentResult(Intent.UNSUPPORTED, 0.7, ["non_health_topic"], False)
