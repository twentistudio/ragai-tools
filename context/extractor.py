"""
Structured health context extraction (§8).

Mengubah kalimat percakapan menjadi struktur terkontrol.

ATURAN UTAMA: **jangan pernah mengarang field**. Apa pun yang tidak disebut
user tetap `None` / list kosong. Ekstraksi dilakukan secara leksikal
(regex + leksikon), bukan oleh LLM, agar tidak ada halusinasi data pasien.
"""

import re
from typing import Dict, List, Optional, Tuple

from ..contracts import HealthContext, Provenance
from ..lexicon import (
    ALL_SYMPTOM_VARIANTS,
    CONDITION_TERMS,
    MEDICATION_TERMS,
    SYMPTOM_LOOKUP,
    find_symptom_variants,
)

# --- durasi -----------------------------------------------------------------

_ID_NUMBERS = {
    "se": 1, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5,
    "enam": 6, "tujuh": 7, "delapan": 8, "sembilan": 9, "sepuluh": 10,
    "sebelas": 11, "dua belas": 12,
}
_EN_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_UNIT_CANON = {
    "jam": "jam", "hour": "jam", "hours": "jam",
    "hari": "hari", "day": "hari", "days": "hari",
    "minggu": "minggu", "week": "minggu", "weeks": "minggu",
    "pekan": "minggu",
    "bulan": "bulan", "month": "bulan", "months": "bulan",
    "tahun": "tahun", "year": "tahun", "years": "tahun",
}

_NUM_WORDS = "|".join(sorted(list(_ID_NUMBERS.keys()) + list(_EN_NUMBERS.keys()), key=len, reverse=True))
_UNITS = "|".join(sorted(_UNIT_CANON.keys(), key=len, reverse=True))

_DURATION_RE = re.compile(
    rf"\b(?:sejak|selama|sudah|udah|kurang lebih|sekitar|for|since|about)?\s*"
    rf"(\d{{1,3}}|{_NUM_WORDS})\s*({_UNITS})\b(?:\s*(?:terakhir|ini|yang lalu|ago))?",
    re.IGNORECASE,
)

_RELATIVE_DURATION = [
    (r"\bsejak\s+kemarin\s+lusa\b", "2 hari"),
    (r"\bsejak\s+kemarin\b", "1 hari"),
    (r"\bkemarin\s+malam\b", "1 hari"),
    (r"\bsejak\s+tadi\s+(pagi|siang|sore|malam)\b", "kurang dari 1 hari"),
    (r"\bbaru\s+(saja|aja)\b", "kurang dari 1 hari"),
    (r"\bseminggu\b", "1 minggu"),
    (r"\bsebulan\b", "1 bulan"),
    (r"\bsetahun\b", "1 tahun"),
    (r"\bbeberapa\s+hari\b", "beberapa hari"),
    (r"\bbeberapa\s+minggu\b", "beberapa minggu"),
    (r"\bbertahun[- ]tahun\b", "bertahun-tahun"),
]

# --- severity / onset / progression -----------------------------------------

_SEVERITY_PATTERNS = [
    ("berat", r"\b(sangat (parah|berat|hebat)|parah sekali|berat sekali|hebat|tak tertahankan|tidak tertahankan|severe|excruciating)\b"),
    ("sedang", r"\b(sedang|lumayan|cukup (parah|mengganggu)|moderate)\b"),
    ("ringan", r"\b(ringan|sedikit|agak|tidak terlalu|mild|slight)\b"),
]
_SEVERITY_SCALE_RE = re.compile(r"\bskala\s*(\d{1,2})\s*(?:dari\s*10|/\s*10)?\b|\b(\d{1,2})\s*/\s*10\b", re.IGNORECASE)

_ONSET_PATTERNS = [
    ("mendadak", r"\b(tiba[- ]tiba|mendadak|sekonyong|sudden(ly)?|abrupt)\b"),
    ("bertahap", r"\b(perlahan|pelan[- ]pelan|bertahap|lama[- ]kelamaan|gradual(ly)?)\b"),
    ("setelah aktivitas", r"\b(setelah|sehabis|habis)\s+(makan|olahraga|minum obat|bekerja|perjalanan)\b"),
]

_PROGRESSION_PATTERNS = [
    ("memburuk", r"\b(makin|semakin|tambah)\s+(parah|buruk|berat|sakit|sering)\b|\bmemburuk\b|\bworsening\b|\bgetting worse\b"),
    ("membaik", r"\b(makin|semakin)\s+(baik|membaik|enak|ringan)\b|\bmembaik\b|\bimproving\b|\bgetting better\b"),
    ("menetap", r"\b(tidak (berubah|membaik)|begitu[- ]begitu saja|menetap|stabil|persistent|unchanged)\b"),
    ("hilang timbul", r"\b(hilang timbul|kadang[- ]kadang|sesekali|intermittent|comes and goes)\b"),
]

# --- alergi & riwayat --------------------------------------------------------

_ALLERGY_RE = re.compile(
    r"\b(?:alergi|alergen|allergic to|allergy to)\s+(?:terhadap\s+|sama\s+|dengan\s+)?"
    r"([a-zA-ZÀ-ɏ][\w\s\-]{1,40}?)(?=[,.;]|\s+(?:dan|atau|tapi|namun|serta|and|or)\b|$)",
    re.IGNORECASE,
)
_NO_ALLERGY_RE = re.compile(r"\b(tidak (ada|punya) alergi|no known allerg|tanpa alergi)\b", re.IGNORECASE)

_HISTORY_RE = re.compile(
    r"\b(?:riwayat|punya|menderita|didiagnosis|pernah kena|history of|diagnosed with)\s+"
    r"([a-zA-ZÀ-ɏ][\w\s\-]{2,40}?)(?=[,.;]|\s+(?:dan|atau|sejak|tapi|namun|and|or|since)\b|$)",
    re.IGNORECASE,
)

_MEDICATION_TAKE_RE = re.compile(
    r"\b(?:minum|mengonsumsi|konsumsi|makan|pakai|memakai|taking|took|using)\s+(?:obat\s+)?"
    r"([a-zA-ZÀ-ɏ][\w\s\-]{2,40}?)(?=[,.;]|\s+(?:dan|atau|tapi|namun|untuk|selama|sejak|and|or|for)\b|$)",
    re.IGNORECASE,
)

_CHIEF_COMPLAINT_HINT = re.compile(
    r"\b(keluhan utama|paling mengganggu|yang paling|terutama|chief complaint)\b", re.IGNORECASE
)


def _normalize_duration(count: str, unit: str) -> str:
    unit_canon = _UNIT_CANON.get(unit.lower(), unit.lower())
    key = count.strip().lower()
    if key.isdigit():
        value = int(key)
    else:
        value = _ID_NUMBERS.get(key) or _EN_NUMBERS.get(key)
        if value is None:
            return ""
    return f"{value} {unit_canon}"


def extract_duration(text: str) -> Optional[str]:
    """Ambil durasi keluhan, mis. '3 hari'. None bila tidak disebut."""
    low = (text or "").lower()
    match = _DURATION_RE.search(low)
    if match:
        normalized = _normalize_duration(match.group(1), match.group(2))
        if normalized:
            return normalized
    for pattern, value in _RELATIVE_DURATION:
        if re.search(pattern, low, re.IGNORECASE):
            return value
    return None


def extract_symptoms(text: str) -> List[str]:
    """Ambil gejala kanonik yang benar-benar disebut (pencocokan kata utuh)."""
    found: List[str] = []
    for variant in find_symptom_variants(text):
        canonical = SYMPTOM_LOOKUP[variant]
        if canonical not in found:
            found.append(canonical)
    return found


def extract_severity(text: str) -> Optional[str]:
    low = (text or "").lower()
    scale = _SEVERITY_SCALE_RE.search(low)
    if scale:
        raw = scale.group(1) or scale.group(2)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
        if value is not None and 0 <= value <= 10:
            if value >= 8:
                return f"berat (skala {value}/10)"
            if value >= 5:
                return f"sedang (skala {value}/10)"
            return f"ringan (skala {value}/10)"
    for label, pattern in _SEVERITY_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE):
            return label
    return None


def extract_onset(text: str) -> Optional[str]:
    low = (text or "").lower()
    for label, pattern in _ONSET_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE):
            return label
    return None


def extract_progression(text: str) -> Optional[str]:
    low = (text or "").lower()
    for label, pattern in _PROGRESSION_PATTERNS:
        if re.search(pattern, low, re.IGNORECASE):
            return label
    return None


def _clean_entity(raw: str) -> str:
    value = re.sub(r"\s+", " ", (raw or "")).strip(" .,;:-")
    return value[:60]


def extract_medications(text: str) -> List[str]:
    low = (text or "").lower()
    found: List[str] = []

    known = [t for t in MEDICATION_TERMS if len(t) > 4 and t in low]
    generic = {"obat", "medicine", "medication", "drug", "dosis", "dose", "dosage",
               "resep", "prescription", "efek samping", "side effect", "vitamin",
               "suplemen", "supplement", "herbal", "tablet", "kapsul", "salep",
               "sirup obat", "injeksi", "suntik", "minum obat", "konsumsi obat",
               "interaksi obat", "drug interaction", "kontraindikasi",
               "contraindication", "antibiotik", "antibiotic"}
    for term in known:
        if term not in generic and term not in found:
            found.append(term)

    for match in _MEDICATION_TAKE_RE.finditer(low):
        candidate = _clean_entity(match.group(1))
        if not candidate or candidate in generic:
            continue
        # hindari menangkap gejala/kata umum
        if candidate in SYMPTOM_LOOKUP or candidate in ("air", "air putih", "makanan"):
            continue
        if candidate not in found:
            found.append(candidate)

    return found


def extract_allergies(text: str) -> List[str]:
    low = (text or "").lower()
    if _NO_ALLERGY_RE.search(low):
        return ["tidak ada alergi yang dilaporkan"]
    found: List[str] = []
    for match in _ALLERGY_RE.finditer(low):
        candidate = _clean_entity(match.group(1))
        if candidate and candidate not in found:
            found.append(candidate)
    return found


def extract_history(text: str) -> List[str]:
    low = (text or "").lower()
    found: List[str] = []
    for match in _HISTORY_RE.finditer(low):
        candidate = _clean_entity(match.group(1))
        if not candidate:
            continue
        if candidate in ("alergi",):
            continue
        if candidate not in found:
            found.append(candidate)
    # kondisi yang disebut langsung juga dicatat sebagai konteks relevan
    for condition in CONDITION_TERMS:
        if condition in low and condition not in found:
            if re.search(rf"\b(riwayat|punya|menderita|didiagnosis)\b[^.]{{0,40}}{re.escape(condition)}", low):
                found.append(condition)
    return found


def extract_health_context(text: str,
                           previous: Optional[HealthContext] = None) -> HealthContext:
    """
    Bangun HealthContext dari satu pesan user.

    Bila `previous` diberikan, hasilnya adalah gabungan (akumulasi) — ini yang
    membuat "Saya demam" + "sudah tiga hari" menjadi
    duration(demam) = 3 hari (§9).
    """
    text = text or ""
    symptoms = extract_symptoms(text)
    duration = extract_duration(text)
    severity = extract_severity(text)
    onset = extract_onset(text)
    progression = extract_progression(text)
    medications = extract_medications(text)
    allergies = extract_allergies(text)
    history = extract_history(text)

    provenance: Dict[str, str] = {}
    for key, value in (
        ("symptoms", symptoms), ("duration", duration), ("severity", severity),
        ("onset", onset), ("progression", progression),
        ("medications", medications), ("allergies", allergies),
        ("relevant_history", history),
    ):
        if value:
            provenance[key] = Provenance.USER_REPORTED.value

    chief_complaint = None
    if symptoms:
        chief_complaint = symptoms[0]
        provenance["chief_complaint"] = (
            Provenance.USER_REPORTED.value if _CHIEF_COMPLAINT_HINT.search(text)
            else Provenance.AI_INFERRED.value
        )

    current = HealthContext(
        chief_complaint=chief_complaint,
        symptoms=symptoms,
        duration=duration,
        severity=severity,
        onset=onset,
        progression=progression,
        associated_symptoms=symptoms[1:] if len(symptoms) > 1 else [],
        medications=medications,
        allergies=allergies,
        relevant_history=history,
        provenance=provenance,
    )

    if previous is None:
        return current

    merged = previous.merge(current)
    # Keluhan utama tetap yang pertama kali dilaporkan.
    if previous.chief_complaint:
        merged.chief_complaint = previous.chief_complaint
    if len(merged.symptoms) > 1:
        merged.associated_symptoms = [s for s in merged.symptoms[1:]]
    return merged


def context_terms(context: HealthContext) -> List[str]:
    """Istilah dari health context untuk memperkaya query retrieval."""
    terms: List[str] = []
    if context.chief_complaint:
        terms.append(context.chief_complaint)
    terms.extend(context.symptoms or [])
    terms.extend(context.associated_symptoms or [])
    terms.extend(context.medications or [])
    terms.extend(context.relevant_history or [])
    seen = set()
    unique = []
    for term in terms:
        low = (term or "").strip().lower()
        if low and low not in seen:
            seen.add(low)
            unique.append(low)
    return unique


def health_context_from_dict(data: Optional[Dict]) -> HealthContext:
    """Rekonstruksi HealthContext dari dict (mis. payload HealthTalk)."""
    data = data or {}
    if not isinstance(data, dict):
        return HealthContext()

    def as_list(key):
        value = data.get(key) or []
        if isinstance(value, str):
            value = [value]
        return [str(v).strip() for v in value if str(v).strip()]

    def as_str(key):
        value = data.get(key)
        if value in (None, "", "unknown", "not_reported"):
            return None
        return str(value).strip()

    return HealthContext(
        chief_complaint=as_str("chief_complaint"),
        symptoms=as_list("symptoms"),
        duration=as_str("duration"),
        severity=as_str("severity"),
        onset=as_str("onset"),
        progression=as_str("progression"),
        associated_symptoms=as_list("associated_symptoms"),
        medications=as_list("medications"),
        allergies=as_list("allergies"),
        relevant_history=as_list("relevant_history"),
        provenance=data.get("provenance") if isinstance(data.get("provenance"), dict) else {},
    )
