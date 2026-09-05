"""
Evidence validation & selection (langkah akhir, ).

Tugas:
1. Jalankan validasi link untuk setiap kandidat evidence (anti-404).
2. Buang evidence yang tidak layak publikasi (termasuk sumber karangan LLM).
3. Tentukan status kecukupan bukti: SUFFICIENT / PARTIAL / INSUFFICIENT_EVIDENCE.

Kalau bukti tidak cukup, pipeline tidak boleh menyuruh LLM menebak, lihat
`api.intelligence.reasoning`.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional, Tuple

from ..contracts import EvidenceItem, EvidenceOrigin, EvidenceStatus
from . import link_validator as lv
from .quality import rank_evidence

logger = logging.getLogger(__name__)

# Ambang kecukupan bukti.
MIN_RELEVANCE_PUBLISHABLE = 0.30
MIN_ITEMS_SUFFICIENT = 2
MIN_TOP_SCORE_SUFFICIENT = 0.55
MIN_TOP_SCORE_PARTIAL = 0.38

# Paper yang sekadar menyebut penyakit yang sama belum menjawab pertanyaannya.
MIN_ASPECT_MATCH_SUFFICIENT = 0.5

# Berapa validasi tautan yang berjalan berbarengan.
VALIDATION_WORKERS = 8

# Lantai kemiripan makna: satu-satunya sinyal yang memisahkan dokumen yang
# hanya berbagi kata umum. Diukur di produksi, kasus tak relevan memuncak di
# 0.19 dan kasus benar terendah 0.33, jadi ambangnya di tengah jurang itu.
MIN_SEMANTIC_RELEVANCE = 0.25

# Origin yang boleh "dipercaya" saat status link tidak bisa dipastikan.
_TRUSTED_ORIGINS = {
    EvidenceOrigin.KNOWLEDGE_BASE,
    EvidenceOrigin.VECTOR_INDEX,
    EvidenceOrigin.USER_SUPPLIED,
}


def validate_links(items: Iterable[EvidenceItem], timeout: float = 5.0) -> List[EvidenceItem]:
    """
    Validasi DOI/URL setiap item dan tulis balik hasilnya ke item.

    Setiap item butuh sampai dua perjalanan jaringan, jadi dikerjakan
    berbarengan: berurutan, delapan referensi memakan sekitar tiga detik.
    Hasilnya di-cache, sehingga ongkosnya hanya untuk DOI yang baru.
    """
    items = list(items or [])
    if not items:
        return []

    if len(items) > 1:
        with ThreadPoolExecutor(max_workers=min(VALIDATION_WORKERS, len(items))) as pool:
            list(pool.map(lambda item: _validate_one(item, timeout), items))
        return items

    _validate_one(items[0], timeout)
    return items


def _validate_one(item: EvidenceItem, timeout: float = 5.0) -> EvidenceItem:
    """Validasi satu item. Dipisah agar bisa dijalankan berbarengan."""
    try:
        result = lv.validate_reference(
            item.doi,
            item.url,
            timeout=timeout,
            trust_on_unknown=item.origin in _TRUSTED_ORIGINS,
        )
    except Exception as exc:  # pragma: no cover - jaring pengaman
        logger.warning("[EVIDENCE] validasi link gagal: %s", exc)
        result = {"doi": "", "url": "", "doi_verified": False,
                  "link_status": lv.STATUS_UNKNOWN}

    item.doi = result["doi"]
    item.url = result["url"]
    item.doi_verified = result["doi_verified"]
    item.link_status = result["link_status"]

    # Judul diambil dari registry: DOI yang terdaftar bisa saja dipasangkan
    # dengan judul milik paper lain, dan pembaca membuka halaman yang berbeda.
    if item.doi and item.doi_verified:
        _apply_registry_metadata(item, timeout=timeout)

    # Sumber karangan LLM yang DOI-nya ternyata benar-benar ada dipromosikan
    # menjadi VERIFIED_REGISTRY.
    if item.origin == EvidenceOrigin.MODEL_SUGGESTED and item.doi_verified:
        item.origin = EvidenceOrigin.VERIFIED_REGISTRY

    return item


def _apply_registry_metadata(item: EvidenceItem, timeout: float = 5.0) -> None:
    """
    Timpa judul (dan metadata terbit) dengan nilai resmi dari registry.

    Bila judul tersimpan ternyata merujuk karya lain, penggantian ini dicatat:
    itu penanda bahwa sumbernya berasal dari data yang tidak tepercaya.
    """
    try:
        metadata = lv.fetch_doi_metadata(item.doi, timeout=timeout)
    except Exception as exc:  # pragma: no cover
        logger.debug("[EVIDENCE] metadata registry gagal untuk %s: %s", item.doi, exc)
        return

    registry_title = (metadata or {}).get("title", "").strip()
    if not registry_title:
        return

    if item.title and not lv.titles_match(item.title, registry_title):
        logger.warning(
            "[EVIDENCE] Judul tidak cocok dengan DOI %s. Tersimpan: %r. "
            "Registry: %r. Judul registry yang dipakai.",
            item.doi, item.title[:90], registry_title[:90],
        )
        item.title_corrected = True

    item.title = registry_title
    if metadata.get("publisher") or metadata.get("container"):
        item.publisher = (metadata.get("container") or metadata.get("publisher") or "").strip()
    if metadata.get("year"):
        item.published_year = metadata["year"]
    if metadata.get("authors") and not item.authors:
        item.authors = metadata["authors"]


def select_evidence(items: Iterable[EvidenceItem],
                    context_terms: Optional[Iterable[str]] = None,
                    limit: int = 5,
                    min_relevance: float = MIN_RELEVANCE_PUBLISHABLE,
                    validate: bool = True,
                    timeout: float = 5.0) -> Tuple[List[EvidenceItem], EvidenceStatus]:
    """
    Pipeline lengkap: validasi -> filter -> skor -> ranking -> potong.

    Returns: (evidence terpilih, status kecukupan)
    """
    candidates = list(items or [])
    if not candidates:
        return [], EvidenceStatus.INSUFFICIENT_EVIDENCE

    if validate:
        candidates = validate_links(candidates, timeout=timeout)

    publishable = [i for i in candidates if i.is_publishable()]
    dropped = len(candidates) - len(publishable)
    if dropped:
        logger.info("[EVIDENCE] %d/%d evidence dibuang (link mati / sumber tak terverifikasi)",
                    dropped, len(candidates))

    # Dokumen yang JUDULNYA jelas membahas topik lain tidak layak disajikan
    # sebagai bukti, berapa pun skor kata kuncinya. Menurunkan bobotnya saja
    # tidak cukup: pembaca tetap membuka paper yang tidak ada kaitannya dengan
    # pertanyaannya, dan itulah keluhan yang paling merusak kepercayaan.
    on_topic = [i for i in publishable if not getattr(i, "off_topic", False)]
    off_topic_count = len(publishable) - len(on_topic)
    if off_topic_count:
        logger.info("[EVIDENCE] %d evidence dibuang karena membahas topik lain",
                    off_topic_count)
    # Bila seluruh kandidat di luar topik, hasil tidak dikosongkan begitu saja;
    # biarkan ambang relevansi di bawah yang memutuskan.
    publishable = on_topic or []

    publishable = _drop_semantically_unrelated(publishable)

    ranked = rank_evidence(publishable, context_terms=context_terms)
    relevant = [i for i in ranked if i.relevance >= min_relevance]

    selected = _dedupe_by_title(relevant)[:limit]
    status = classify_sufficiency(selected)

    # Konsistensi kontrak: bukti yang dinyatakan tidak memadai tidak
    # menerbitkan daftar sumber, response akan bertentangan dengan dirinya
    # sendiri ("bukti tidak ditemukan" tetapi ada sumber terlampir).
    if status == EvidenceStatus.INSUFFICIENT_EVIDENCE:
        return [], status

    return selected, status


def _dedupe_by_title(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Satu judul cukup sekali.

    Knowledge base memuat beberapa paper dengan judul sama tetapi DOI berbeda
    (versi bab buku dan versi artikel, atau terbitan ulang). Keduanya sah, tapi
    menampilkan keduanya memakan dua slot referensi untuk satu bacaan yang sama
    dan membuat daftar rujukan tampak lebih tebal daripada isinya. Yang
    peringkatnya lebih tinggi dipertahankan, karena daftar sudah terurut.
    """
    seen = set()
    kept = []
    for item in items:
        marker = " ".join((item.title or "").lower().split())
        if marker and marker in seen:
            continue
        if marker:
            seen.add(marker)
        kept.append(item)
    if len(kept) != len(items):
        logger.info("[EVIDENCE] %d evidence dilewati karena judulnya kembar",
                    len(items) - len(kept))
    return kept


def _drop_semantically_unrelated(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Buang dokumen yang maknanya terlalu jauh dari pertanyaan.

    Penyaringan hanya dijalankan bila penilaian semantik memang tersedia.
    Ketika embedding dimatikan atau penyedia sedang gagal, seluruh kandidat
    bernilai 0.0; menerapkan lantai pada keadaan itu akan mengosongkan mesin
    sepenuhnya, bukan menyaringnya. Karena itu nol di semua kandidat dibaca
    sebagai "sinyal tidak ada", bukan sebagai "semuanya tidak relevan".
    """
    if not items:
        return items

    scored = [i for i in items if (i.semantic_relevance or 0.0) > 0.0]
    if not scored:
        logger.info("[EVIDENCE] penilaian semantik tidak tersedia; lantai kemiripan dilewati")
        return items

    related = [i for i in items
               if (i.semantic_relevance or 0.0) >= MIN_SEMANTIC_RELEVANCE]
    if len(related) != len(items):
        logger.info("[EVIDENCE] %d evidence dibuang karena kemiripan makna di bawah %.2f",
                    len(items) - len(related), MIN_SEMANTIC_RELEVANCE)
    return related


def classify_sufficiency(selected: List[EvidenceItem]) -> EvidenceStatus:
    """Tentukan status kecukupan bukti."""
    if not selected:
        return EvidenceStatus.INSUFFICIENT_EVIDENCE

    top = selected[0].relevance
    best_aspect = max(
        (getattr(item, "aspect_match", 1.0) for item in selected), default=1.0
    )
    aspect_ok = best_aspect >= MIN_ASPECT_MATCH_SUFFICIENT

    if (len(selected) >= MIN_ITEMS_SUFFICIENT
            and top >= MIN_TOP_SCORE_SUFFICIENT
            and aspect_ok):
        return EvidenceStatus.SUFFICIENT
    if top >= MIN_TOP_SCORE_PARTIAL:
        return EvidenceStatus.PARTIAL
    return EvidenceStatus.INSUFFICIENT_EVIDENCE
