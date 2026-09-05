"""
Evidence quality & ranking.

Skor akhir dihitung dari 5 dimensi:
    semantic_relevance   - seberapa dekat evidence dengan query (dari retriever)
    source_quality       - kualitas penerbit/jenis sumber
    publication_recency  - kebaruan publikasi
    evidence_type        - jenis studi (meta-analisis > RCT > observational > ...)
    context_match        - kecocokan dengan health context user (gejala, obat)

Modul ini sengaja dipisah agar bobot/heuristik bisa diperbaiki tanpa
menyentuh pipeline retrieval maupun generation.
"""

import datetime
import logging
import re
from typing import Any, Dict, Iterable, List, Optional
from .. import config

logger = logging.getLogger(__name__)

# Bobot default. Bisa dioverride lewat settings.EVIDENCE_SCORE_WEIGHTS.
# Dokumen yang membahas topik tetapi tidak membahas aspek yang ditanyakan
# hanya mempertahankan sebagian skornya. Menyebut nama penyakit tidak
# menjadikan sebuah paper sebagai jawaban atas pertanyaan tentang penyakit itu.
ASPECT_FLOOR = 0.45

DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic_relevance": 0.45,
    "source_quality": 0.15,
    "publication_recency": 0.10,
    "evidence_type": 0.15,
    "context_match": 0.15,
}

# Prefix DOI penerbit akademik tepercaya -> skor kualitas.
_PUBLISHER_QUALITY = {
    "10.1056/": 0.98,   # NEJM
    "10.1016/": 0.90,   # Elsevier
    "10.1001/": 0.95,   # JAMA
    "10.1136/": 0.90,   # BMJ
    "10.1038/": 0.93,   # Nature
    "10.1126/": 0.93,   # Science
    "10.1371/": 0.88,   # PLOS
    "10.1186/": 0.85,   # BMC
    "10.3389/": 0.82,   # Frontiers
    "10.1002/": 0.85,   # Wiley (incl. Cochrane)
    "10.1111/": 0.85,   # Wiley
    "10.1093/": 0.88,   # Oxford UP
    "10.1007/": 0.85,   # Springer
    "10.1097/": 0.85,   # LWW
    "10.1017/": 0.84,   # Cambridge UP
    "10.1073/": 0.92,   # PNAS
    "10.1101/": 0.60,   # bioRxiv/medRxiv preprint
    "10.48550/": 0.55,  # arXiv preprint
}

_SOURCE_TYPE_QUALITY = {
    "journal": 0.85,
    "government": 0.80,
    "organization": 0.70,
    "website": 0.45,
    "news": 0.25,
    "other": 0.40,
}

# Jenis bukti -> skor (evidence hierarchy).
_EVIDENCE_TYPE_PATTERNS = [
    (0.98, r"\b(cochrane|meta[- ]analysis|metaanalisis|meta analisis|systematic review|tinjauan sistematis)\b"),
    (0.90, r"\b(randomi[sz]ed controlled trial|randomi[sz]ed[- ]control|\brct\b|uji klinis acak)\b"),
    (0.78, r"\b(clinical trial|uji klinis|double[- ]blind|tersamar ganda)\b"),
    (0.70, r"\b(cohort study|studi kohort|prospective study|longitudinal study)\b"),
    (0.62, r"\b(case[- ]control|kasus kontrol|cross[- ]sectional|potong lintang)\b"),
    (0.55, r"\b(observational|observasional|survey|survei)\b"),
    (0.45, r"\b(guideline|pedoman|consensus statement|rekomendasi)\b"),
    (0.35, r"\b(case report|laporan kasus|case series)\b"),
    (0.30, r"\b(in vitro|animal study|studi hewan|mouse model|tikus)\b"),
    (0.25, r"\b(narrative review|tinjauan pustaka|editorial|opinion|opini)\b"),
]

_PREPRINT_RE = re.compile(r"\b(preprint|biorxiv|medrxiv|arxiv|ssrn)\b", re.IGNORECASE)


def _get_weights() -> Dict[str, float]:
    override = config.get_dict("EVIDENCE_SCORE_WEIGHTS")
    if not override:
        return dict(DEFAULT_WEIGHTS)
    weights = dict(DEFAULT_WEIGHTS)
    for key, value in override.items():
        if key in weights:
            try:
                weights[key] = float(value)
            except (TypeError, ValueError):
                continue
    return weights


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def score_source_quality(doi: str = "", source_type: str = "journal",
                         publisher: str = "", url: str = "") -> float:
    """Kualitas sumber berdasarkan prefix DOI, tipe, dan penerbit."""
    doi = (doi or "").strip().lower()
    for prefix, score in _PUBLISHER_QUALITY.items():
        if doi.startswith(prefix):
            return score

    base = _SOURCE_TYPE_QUALITY.get((source_type or "other").lower(), 0.40)

    haystack = f"{publisher} {url}".lower()
    if any(tok in haystack for tok in ("who.int", "cdc.gov", "nih.gov", "kemkes.go.id", "nice.org.uk")):
        return max(base, 0.88)
    if any(tok in haystack for tok in ("pubmed", "pmc.ncbi", "cochrane")):
        return max(base, 0.90)
    if doi:
        # DOI valid tapi penerbit tidak dikenal -> tetap di atas website biasa.
        return max(base, 0.65)
    return base


def score_recency(published_year: Optional[int], now_year: Optional[int] = None) -> float:
    """
    Kebaruan publikasi. <=2 tahun -> 1.0, lalu meluruh; >25 tahun -> 0.15.
    Tahun tidak diketahui -> 0.5 (netral, tidak menghukum).
    """
    if not published_year:
        return 0.5
    try:
        year = int(published_year)
    except (TypeError, ValueError):
        return 0.5
    now_year = now_year or datetime.date.today().year
    if year > now_year + 1:  # tahun tidak masuk akal
        return 0.5
    age = max(0, now_year - year)
    if age <= 2:
        return 1.0
    if age >= 25:
        return 0.15
    return _clamp(1.0 - ((age - 2) / 23.0) * 0.85)


def score_evidence_type(text: str = "", title: str = "") -> float:
    """Perkirakan jenis studi dari judul + abstrak."""
    haystack = f"{title or ''} {text or ''}".lower()
    if not haystack.strip():
        return 0.5
    for score, pattern in _EVIDENCE_TYPE_PATTERNS:
        if re.search(pattern, haystack, re.IGNORECASE):
            if _PREPRINT_RE.search(haystack) and score > 0.6:
                return score * 0.8  # preprint belum peer-review
            return score
    return 0.5


def score_context_match(text: str, title: str, context_terms: Iterable[str]) -> float:
    """
    Seberapa banyak istilah dari health context user muncul di evidence.

    Istilah konteks berasal dari kalimat pengguna, jadi berbahasa Indonesia,
    sedangkan hampir seluruh knowledge base berbahasa Inggris. Mencocokkan apa
    adanya membuat komponen ini nyaris selalu bernilai nol, sehingga setiap
    jurnal berbahasa Inggris kehilangan bobotnya tanpa alasan yang sebenarnya.
    Padanan Inggris tiap istilah ikut dihitung: satu istilah dianggap cocok
    bila bentuk mana pun darinya muncul.
    """
    terms = [t.strip().lower() for t in (context_terms or []) if t and str(t).strip()]
    if not terms:
        return 0.0

    haystack = f"{title or ''} {text or ''}".lower()
    unique = set(terms)

    hits = 0
    for term in unique:
        forms = {term}
        try:
            from ..lexicon import bilingual_variants

            forms |= {v.lower() for v in bilingual_variants(term)}
        except Exception:  # pragma: no cover - leksikon tidak wajib
            pass
        if any(form in haystack for form in forms):
            hits += 1

    return _clamp(hits / float(len(unique)))


def compute_evidence_score(item, context_terms: Optional[Iterable[str]] = None) -> float:
    """
    Hitung dan tempelkan skor gabungan ke sebuah EvidenceItem.
    Mengembalikan skor akhir (0..1) dan memutakhirkan atribut item.
    """
    weights = _get_weights()

    item.source_quality = score_source_quality(
        doi=item.doi, source_type=item.source_type,
        publisher=item.publisher, url=item.url,
    )
    item.publication_recency = score_recency(item.published_year)
    item.evidence_type_score = score_evidence_type(item.snippet, item.title)
    item.context_match = score_context_match(item.snippet, item.title, context_terms or [])
    item.semantic_relevance = _clamp(item.semantic_relevance)

    total_weight = sum(weights.values()) or 1.0
    score = (
        weights["semantic_relevance"] * item.semantic_relevance
        + weights["source_quality"] * item.source_quality
        + weights["publication_recency"] * item.publication_recency
        + weights["evidence_type"] * item.evidence_type_score
        + weights["context_match"] * item.context_match
    ) / total_weight

    # Faktor aspek: 1.0 bila seluruh aspek yang ditanyakan dibahas, turun ke
    # ASPECT_FLOOR bila tidak satu pun dibahas.
    aspect = _clamp(getattr(item, "aspect_match", 1.0))
    score *= ASPECT_FLOOR + (1.0 - ASPECT_FLOOR) * aspect

    item.relevance = round(_clamp(score), 4)
    return item.relevance


def rank_evidence(items: List[Any], context_terms: Optional[Iterable[str]] = None,
                  limit: Optional[int] = None) -> List[Any]:
    """Skor + urutkan + potong. Tidak mengubah list input."""
    scored = list(items or [])
    for item in scored:
        compute_evidence_score(item, context_terms=context_terms)
    scored.sort(key=lambda i: getattr(i, "relevance", 0.0), reverse=True)
    if limit is not None:
        scored = scored[:limit]
    return scored
