"""
Evidence retrieval (§10).

Pipeline:
    Query -> Normalisasi -> Ekstraksi konsep -> Retrieval semantik/leksikal
          -> Filter metadata -> (re-ranking & seleksi ada di modul evidence)

Sumber pengetahuan yang dipakai — SEMUANYA yang sudah ada, tidak ada yang
diganti (§10, §22):

    1. `JournalArticle`  : knowledge base jurnal yang diinput admin Healthify.
    2. `Source`          : sumber yang sudah pernah terkait dengan klaim.
    3. tabel embeddings  : indeks pgvector milik pipeline training (opsional,
                           dipakai bila modul training bisa diimpor).

Yang TIDAK pernah dilakukan modul ini: meminta LLM mengarang sumber.
Setiap EvidenceItem yang keluar dari sini berasal dari baris nyata di
knowledge base, lengkap dengan id-nya, sehingga bisa ditelusuri (§14).
"""

import logging
import math
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from ..contracts import EvidenceItem, EvidenceOrigin
from ..lexicon import find_aspects
from .. import runtime
from .concepts import (
    build_aspect_groups,
    extract_conditions,
    extract_health_concepts,
    build_embedding_query,
    build_search_term_groups,
    build_search_terms,
    normalize_query,
    tokenize,
)

logger = logging.getLogger(__name__)

# Seberapa luas mesin membaca sebelum memutuskan, bukan berapa referensi yang
# disajikan. Membaca lebih luas hampir gratis karena skoring leksikal murah.
DEFAULT_CANDIDATE_LIMIT = 80

# Batas baris yang diskor per sumber, dibuat longgar. Memotong kandidat lebih
# awal berdasarkan tanggal membuat relevansi ditentukan oleh kebaruan.
MAX_SCORED_CANDIDATES = 500

# Cosine similarity antar teks kesehatan selalu tinggi, jadi dikalibrasi
# terhadap lantainya. Terukur: ada jawaban 0,435-0,657; tidak ada 0,303-0,414.
SEMANTIC_FLOOR = 0.35
SEMANTIC_WEIGHT = 0.5

# Cocok kata kunci tanpa dukungan semantik hanya menyisakan sebagian skor:
# boleh muncul sebagai bukti terbatas, tidak cukup untuk dinyatakan memadai.
NO_SEMANTIC_SUPPORT_PENALTY = 0.65

# Nilai untuk dokumen yang judulnya membahas topik lain, sekalipun abstraknya
# memuat kata kunci yang sama.
OFF_TOPIC_TITLE_SCORE = 0.25

# Batas fokus judul. "Tuberculosis treatment adherence in the era of COVID-19"
# menyebut COVID-19, tetapi pokoknya tuberkulosis.
MIN_TITLE_FOCUS = 0.5

# Aspek di judul atau keywords menandakan pokok bahasan; di badan abstrak jauh
# lebih lemah karena kata seperti "effective" ada di hampir semua abstrak.
ASPECT_BODY_CREDIT = 0.5
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# --------------------------------------------------------------------------
# Pencocokan leksikal (bekerja di PostgreSQL maupun SQLite)
# --------------------------------------------------------------------------

def _as_groups(terms) -> List[Dict[str, Any]]:
    """
    Normalkan masukan menjadi kelompok berbobot.

    Menerima: daftar istilah datar (`["demam", ...]`), daftar kelompok varian
    (`[["demam", "fever"], ...]`), atau daftar dict berbobot dari
    `build_search_term_groups`.
    """
    groups: List[Dict[str, Any]] = []
    for term in terms or []:
        if isinstance(term, dict):
            variants = [str(t).strip().lower() for t in (term.get("variants") or []) if str(t).strip()]
            if variants:
                groups.append({
                    "variants": variants,
                    "weight": float(term.get("weight", 1.0)),
                })
        elif isinstance(term, (list, tuple, set)):
            variants = [str(t).strip().lower() for t in term if str(t).strip()]
            if variants:
                groups.append({
                    "variants": variants,
                    "weight": 1.0 + min(len(variants[0].split()), 3) * 0.5,
                })
        else:
            value = str(term).strip().lower()
            if value:
                groups.append({
                    "variants": [value],
                    "weight": 1.0 + min(len(value.split()), 3) * 0.5,
                })
    return groups


def _topical_score(aspect_groups, query_subjects, title, body, keywords=""):
    """
    Gabungkan dua ukuran relevansi yang mengukur hal berbeda:

      fokus judul     : apakah dokumen ini MEMBAHAS topik yang ditanyakan, atau
                        hanya menyebutnya sambil lalu;
      cakupan aspek   : apakah isinya menjawab hal yang ditanyakan.

    Keduanya tidak boleh digabung dengan `min`. Paper berjudul "Typhoid Fever
    Diagnosis" yang abstraknya menjelaskan penularan tetap menjawab pertanyaan
    tentang penularan, meski judulnya menyebut aspek lain.

    Namun judul yang jelas membahas topik lain tetap menjadi penentu: isi yang
    kebetulan memuat kata kunci tidak menyelamatkannya.

    Returns: (skor, off_topic)
    """
    focus = _title_focus(title, query_subjects or set())

    # 0.0 adalah penanda "judul membahas topik lain sama sekali", berbeda dari
    # rasio kecil seperti 1/4 yang berarti judul menyebut beberapa hal dan
    # salah satunya memang yang ditanyakan.
    if focus < MIN_TITLE_FOCUS:
        return OFF_TOPIC_TITLE_SCORE, True

    coverage = _aspect_coverage(aspect_groups, title, body, keywords)
    return 0.4 * focus + 0.6 * coverage, False


def _title_focus(title: str, query_subjects: set) -> float:
    """
    Seberapa besar bagian JUDUL dokumen yang memang membahas hal yang ditanyakan.

    Kecocokan kata kunci tidak membedakan dokumen yang membahas sebuah topik
    dari yang hanya menyebutnya sambil lalu.

    Diukur terbalik: dari seluruh topik yang disebut judul, berapa bagian yang
    juga ditanyakan. Judul tanpa topik yang dikenali bernilai 1.0 (tidak ada
    dasar menghukum), judul yang seluruhnya membahas hal lain bernilai 0.0.
    """
    if not query_subjects:
        return 1.0

    # Hanya penyakit/gejala yang dibandingkan. Kosakata umum dan aspek tidak
    # menentukan "dokumen ini tentang apa".
    title_subjects = _collapse_nested(set(extract_conditions(title or "")))
    if not title_subjects:
        return 1.0

    # Pencocokan sadar hierarki: "demam berdarah" pada judul menjawab
    # pertanyaan tentang "demam". Irisan himpunan biasa akan menganggap
    # keduanya berbeda dan menghukum dokumen yang justru paling tepat.
    matched = sum(
        1 for subject in title_subjects
        if any(subject in q or q in subject for q in query_subjects)
    )
    if not matched:
        return 0.0  # penanda: judul membahas hal lain sama sekali

    return matched / float(len(title_subjects))


def _collapse_nested(subjects: set) -> set:
    """
    Buang istilah yang sudah tercakup istilah lain yang lebih spesifik.

    Ekstraksi konsep menghasilkan "demam" sekaligus "demam berdarah" untuk judul
    yang sama. Menghitung keduanya menggelembungkan penyebut dan membuat judul
    yang seluruhnya pada topik tampak setengah melenceng.
    """
    kept = set()
    for subject in sorted(subjects, key=len, reverse=True):
        if not any(subject != other and subject in other for other in kept):
            kept.add(subject)
    return kept


SHORT_VARIANT_MAX = 4


@lru_cache(maxsize=2048)
def _boundary_pattern(variant: str):
    return re.compile(r"\b" + re.escape(variant) + r"\b")


def _variant_in(variant: str, text: str) -> bool:
    """
    Cocokkan satu varian istilah pada teks.

    Istilah pendek satu kata dicocokkan dengan batas kata. Pencocokan substring
    mentah membuat istilah sependek "tes" cocok di dalam "diabetes" dan "gula"
    di dalam "regulation", sehingga dokumen yang sama sekali tidak membahas
    pertanyaan mendapat skor leksikal penuh. Istilah yang lebih panjang atau
    berupa frasa tetap dicocokkan sebagai substring, karena di sana kecocokan
    parsial justru berguna ("dengue" pada "dengue fever").
    """
    if not variant:
        return False
    if len(variant) <= SHORT_VARIANT_MAX and " " not in variant:
        return bool(_boundary_pattern(variant).search(text))
    return variant in text


def _aspect_coverage(aspect_groups, title: str, body: str, keywords: str = "") -> float:
    """
    Berapa bagian dari aspek yang ditanyakan benar-benar dibahas dokumen ini.

    Kemunculan di judul atau keywords dihitung penuh; di badan abstrak hanya
    sebagian, karena istilah aspek yang generik muncul di hampir semua abstrak
    medis dan tidak menandakan apa pun.

    Mengembalikan 1.0 bila pertanyaan tidak menyebut aspek tertentu (tidak ada
    yang bisa dinilai), sehingga dokumen tidak dihukum tanpa dasar.
    """
    groups = [g for g in (aspect_groups or []) if g]
    if not groups:
        return 1.0

    strong = f"{title or ''} {keywords or ''}".lower()
    weak = (body or "").lower()

    credit = 0.0
    for variants in groups:
        if any(_variant_in(variant, strong) for variant in variants):
            credit += 1.0
        elif any(_variant_in(variant, weak) for variant in variants):
            credit += ASPECT_BODY_CREDIT

    return credit / float(len(groups))


def _lexical_score(terms, title: str, body: str, keywords: str = "") -> float:
    """
    Skor kecocokan leksikal sederhana, terinspirasi TF-IDF ringan:
      - kecocokan di judul berbobot 3x,
      - kecocokan di keywords berbobot 2x,
      - istilah yang lebih panjang/spesifik berbobot lebih besar.

    `terms` boleh berupa daftar istilah datar, atau daftar **kelompok varian
    bilingual** (mis. `["demam berdarah", "dengue", "dengue fever"]`). Satu
    kelompok dihitung sebagai satu istilah: cukup salah satu varian cocok, dan
    bobotnya diambil dari varian terpanjang yang cocok.
    """
    groups = _as_groups(terms)
    if not groups:
        return 0.0

    title_low = (title or "").lower()
    body_low = (body or "").lower()
    kw_low = (keywords or "").lower()

    total_weight = 0.0
    matched_weight = 0.0

    for group in groups:
        weight = group["weight"]
        total_weight += weight

        best_hit = 0.0
        for variant in group["variants"]:
            if _variant_in(variant, title_low):
                best_hit = 3.0
                break
            if _variant_in(variant, kw_low):
                best_hit = max(best_hit, 2.0)
            elif _variant_in(variant, body_low):
                best_hit = max(best_hit, 1.0)
        matched_weight += weight * best_hit

    if total_weight <= 0:
        return 0.0

    raw = matched_weight / (total_weight * 3.0)  # 3.0 = bobot maksimum per term
    return max(0.0, min(1.0, raw))


def _calibrated_similarity(sim: float) -> float:
    """
    Buang "lantai" cosine similarity sehingga hanya kemiripan yang benar-benar
    di atas rata-rata yang berkontribusi.
    """
    if sim <= SEMANTIC_FLOOR:
        return 0.0
    return (sim - SEMANTIC_FLOOR) / (1.0 - SEMANTIC_FLOOR)


def _blend_scores(lexical: float, similarity: Optional[float]) -> float:
    """
    Gabungkan skor leksikal dan semantik.

    - Tanpa embedding, skor leksikal dipakai apa adanya.
    - Dengan dukungan semantik, skor bisa naik.
    - Tanpa dukungan semantik padahal embedding tersedia, skor ditahan: itu
      tanda dokumen hanya berbagi kata kunci, bukan pokok bahasan.
    """
    if similarity is None:
        return lexical

    adjusted = _calibrated_similarity(similarity)
    if adjusted <= 0:
        return lexical * NO_SEMANTIC_SUPPORT_PENALTY

    blended = (1.0 - SEMANTIC_WEIGHT) * lexical + SEMANTIC_WEIGHT * adjusted
    return max(lexical, blended)


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a or [])
    b = list(b or [])
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def _year_of(value) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "year"):
        try:
            return int(value.year)
        except Exception:
            return None
    match = _YEAR_RE.search(str(value))
    return int(match.group(0)) if match else None


# --------------------------------------------------------------------------
# Sumber 1: JournalArticle (knowledge base kurasi admin)
# --------------------------------------------------------------------------

def _candidate_filter(terms):
    """
    Bangun Q filter untuk mempersempit kandidat sebelum skoring.
    Seluruh varian bilingual ikut dicari agar jurnal berbahasa Inggris
    tetap terjaring oleh query Bahasa Indonesia.
    """
    from django.db.models import Q
    query = Q()
    used = 0
    for group in _as_groups(terms):
        matched_any = False
        for variant in group["variants"]:
            if len(variant) < 3:
                continue
            query |= (Q(title__icontains=variant)
                      | Q(abstract__icontains=variant)
                      | Q(keywords__icontains=variant))
            matched_any = True
        if matched_any:
            used += 1
        if used >= 8:
            break
    return query if used else None


def retrieve_from_journals(terms: List[str], limit: int = DEFAULT_CANDIDATE_LIMIT,
                           query_embedding: Optional[List[float]] = None,
                           aspect_groups: Optional[List[List[str]]] = None,
                           query_subjects: Optional[set] = None) -> List[EvidenceItem]:
    """Ambil kandidat dari tabel JournalArticle."""
    try:
        JournalArticle = runtime.model("JournalArticle")
    except Exception as exc:  # pragma: no cover
        logger.warning("[RETRIEVAL] JournalArticle tidak tersedia: %s", exc)
        return []

    try:
        queryset = JournalArticle.objects.all()
        condition = _candidate_filter(terms)
        if condition is not None:
            queryset = queryset.filter(condition)
        # Skor SELURUH kandidat yang lolos penyaringan kata kunci, bukan hanya
        # yang terbaru — kebaruan bukan proksi relevansi.
        rows = list(queryset[:MAX_SCORED_CANDIDATES])
    except Exception as exc:
        logger.warning("[RETRIEVAL] query JournalArticle gagal: %s", exc)
        return []

    items: List[EvidenceItem] = []
    for row in rows:
        score = _lexical_score(terms, row.title, row.abstract, row.keywords or "")

        topical, off_topic = _topical_score(
            aspect_groups, query_subjects, row.title, row.abstract, row.keywords or ""
        )

        similarity = None
        if query_embedding and row.embedding:
            try:
                import json
                similarity = _cosine(query_embedding, json.loads(row.embedding))
            except Exception:
                similarity = None
        score = _blend_scores(score, similarity)

        if score <= 0:
            continue

        items.append(EvidenceItem(
            chunk_id=f"journal:{row.id}",
            source_id=f"journal:{row.id}",
            title=(row.title or "").strip(),
            snippet=(row.abstract or "").strip()[:1200],
            doi=(row.doi or "").strip(),
            url=(row.url or "").strip(),
            authors=(row.authors or "").strip(),
            publisher=(row.publisher or row.journal_name or "").strip(),
            published_year=_year_of(row.published_date),
            source_type="journal",
            origin=EvidenceOrigin.KNOWLEDGE_BASE,
            semantic_relevance=score,
            aspect_match=topical,
            off_topic=off_topic,
        ))

    items.sort(key=lambda i: i.semantic_relevance, reverse=True)
    return items[:limit]


# --------------------------------------------------------------------------
# Sumber 2: Source (sumber yang sudah tertaut ke klaim terverifikasi)
# --------------------------------------------------------------------------

def retrieve_from_sources(terms: List[str], limit: int = DEFAULT_CANDIDATE_LIMIT,
                          aspect_groups: Optional[List[List[str]]] = None,
                          query_subjects: Optional[set] = None) -> List[EvidenceItem]:
    """Ambil kandidat dari tabel Source + excerpt ClaimSource."""
    try:
        from django.db.models import Q
        ClaimSource = runtime.model("ClaimSource")
    except Exception as exc:  # pragma: no cover
        logger.warning("[RETRIEVAL] ClaimSource tidak tersedia: %s", exc)
        return []

    condition = Q()
    used = 0
    for group in _as_groups(terms):
        matched_any = False
        for variant in group["variants"]:
            if len(variant) < 3:
                continue
            condition |= (Q(source__title__icontains=variant)
                          | Q(excerpt__icontains=variant))
            matched_any = True
        if matched_any:
            used += 1
        if used >= 8:
            break
    if not used:
        return []

    try:
        rows = list(
            ClaimSource.objects.filter(condition)
            .select_related("source")
            .order_by("-relevance_score")[:MAX_SCORED_CANDIDATES]
        )
    except Exception as exc:
        logger.warning("[RETRIEVAL] query ClaimSource gagal: %s", exc)
        return []

    seen: set = set()
    items: List[EvidenceItem] = []
    for row in rows:
        source = row.source
        if source is None or source.id in seen:
            continue
        seen.add(source.id)

        # Sumber tanpa DOI maupun URL tidak dapat ditelusuri pembaca. Baris
        # semacam ini muncul setelah `audit_source_links` membuang tautan yang
        # terbukti karangan; judulnya ikut tidak dapat dipercaya, jadi tidak
        # boleh dimunculkan kembali sebagai bukti.
        if not (source.doi or "").strip() and not (source.url or "").strip():
            continue

        score = _lexical_score(terms, source.title, row.excerpt or "")
        if score <= 0:
            continue

        topical, off_topic = _topical_score(
            aspect_groups, query_subjects, source.title, row.excerpt or ""
        )

        items.append(EvidenceItem(
            chunk_id=f"claimsource:{row.id}",
            source_id=f"source:{source.id}",
            title=(source.title or "").strip(),
            snippet=(row.excerpt or "").strip()[:1200],
            doi=(source.doi or "").strip(),
            url=(source.url or "").strip(),
            authors=(source.authors or "").strip(),
            publisher=(source.publisher or "").strip(),
            published_year=_year_of(source.published_date),
            source_type=source.source_type or "journal",
            origin=EvidenceOrigin.KNOWLEDGE_BASE,
            semantic_relevance=score,
            aspect_match=topical,
            off_topic=off_topic,
        ))

    items.sort(key=lambda i: i.semantic_relevance, reverse=True)
    return items[:limit]


# --------------------------------------------------------------------------
# Sumber 3: indeks vektor pipeline training (pgvector) — opsional
# --------------------------------------------------------------------------

def _training_retrieval_available() -> bool:
    try:
        available = runtime.service("training_modules_available")
        return bool(available and available())
    except Exception:
        return False


def embed_query(text: str) -> Optional[List[float]]:
    """
    Embed query lewat penyedia yang tersedia (OpenAI / Gemini / pipeline training).

    Tidak lagi terikat pada modul training: embedding `JournalArticle` yang
    tersimpan di database Django tetap bisa dicocokkan secara semantik meski
    pipeline training tidak terpasang di lingkungan produksi.
    """
    if not text:
        return None
    try:
        from .embeddings import embed_text
        return embed_text(text)
    except Exception as exc:
        logger.info("[RETRIEVAL] embedding query tidak tersedia: %s", exc)
        return None


def retrieve_from_vector_index(query: str, terms: List[str], k: int = 10,
                               query_embedding: Optional[List[float]] = None,
                               aspect_groups: Optional[List[List[str]]] = None) -> List[EvidenceItem]:
    """
    Ambil kandidat dari indeks pgvector milik pipeline training.
    Gagal secara diam-diam bila pipeline/DB tidak tersedia — Healthify tetap
    berjalan dengan knowledge base Django saja.
    """
    if not _training_retrieval_available():
        return []

    embedding = query_embedding or embed_query(query)
    if not embedding:
        return []

    try:
        import sys
        scripts_dir = runtime.service("training_scripts_dir")
        TRAINING_SCRIPTS_DIR = scripts_dir() if scripts_dir else None
        if TRAINING_SCRIPTS_DIR and str(TRAINING_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(TRAINING_SCRIPTS_DIR))
        from prompt_and_verify import retrieve_neighbors_from_db  # type: ignore
        neighbors = retrieve_neighbors_from_db(embedding, k=k)
    except Exception as exc:
        logger.info("[RETRIEVAL] indeks vektor tidak tersedia: %s", exc)
        return []

    items: List[EvidenceItem] = []
    for neighbor in neighbors or []:
        text = (neighbor.get("text") or "").strip()
        distance = neighbor.get("distance")
        try:
            sim = 1.0 / (1.0 + float(distance)) if distance is not None else 0.5
        except (TypeError, ValueError):
            sim = 0.5
        lexical = _lexical_score(terms, "", text)
        items.append(EvidenceItem(
            chunk_id=str(neighbor.get("doc_id") or neighbor.get("safe_id") or ""),
            source_id=str(neighbor.get("safe_id") or neighbor.get("doc_id") or ""),
            title=str(neighbor.get("safe_id") or "").replace("_", " ").strip(),
            snippet=text[:1200],
            doi=(neighbor.get("doi") or "").strip(),
            url="",
            source_type="journal",
            origin=EvidenceOrigin.VECTOR_INDEX,
            semantic_relevance=_blend_scores(lexical, sim),
            aspect_match=_aspect_coverage(aspect_groups, "", text),
        ))
    return items


# --------------------------------------------------------------------------
# Orkestrasi
# --------------------------------------------------------------------------

def _dedupe(items: List[EvidenceItem]) -> List[EvidenceItem]:
    """Buang duplikat berdasarkan DOI, lalu judul ternormalisasi."""
    seen_doi: set = set()
    seen_title: set = set()
    unique: List[EvidenceItem] = []
    for item in items:
        doi_key = (item.doi or "").strip().lower()
        title_key = normalize_query(item.title)[:120]
        if doi_key and doi_key in seen_doi:
            continue
        if title_key and title_key in seen_title:
            continue
        if doi_key:
            seen_doi.add(doi_key)
        if title_key:
            seen_title.add(title_key)
        unique.append(item)
    return unique


def rescore_for_query(items, query: str, extra_terms=None):
    """
    Nilai ulang sekumpulan bukti terhadap pertanyaan yang sedang diajukan.

    Dipakai untuk bukti yang dibangun ulang dari ingatan percakapan. Baris yang
    dimuat dari basis data tidak membawa skor apa pun, sehingga tanpa penilaian
    ulang seluruhnya jatuh di bawah ambang relevansi dan jawaban lanjutan
    kehilangan rujukannya justru pada pembahasan yang sama.

    Memakai fungsi penilaian yang sama dengan retrieval biasa, kemiripan makna
    termasuk. Menghitung leksikalnya saja memberi skor lebih rendah daripada
    saat jurnal itu pertama kali terpilih, dan ambang kecukupan lalu menolaknya.
    """
    terms = list(extra_terms or []) + extract_health_concepts(query)
    groups = build_search_term_groups(" ".join([query] + terms))
    aspect_groups = build_aspect_groups(query)
    subjects = set(extract_conditions(query))

    query_embedding = None
    try:
        query_embedding = embed_query(build_embedding_query(query, terms))
    except Exception as exc:  # pragma: no cover
        logger.warning("[RETRIEVAL] embedding untuk penilaian ulang gagal: %s", exc)

    stored = _stored_embeddings(items) if query_embedding else {}

    for item in items or []:
        title = item.title or ""
        body = item.snippet or ""
        lexical = _lexical_score(groups, title, body, "")
        topical, off_topic = _topical_score(aspect_groups, subjects, title, body, "")

        similarity = None
        vector = stored.get(item.source_id)
        if query_embedding and vector:
            try:
                similarity = _cosine(query_embedding, vector)
            except Exception:
                similarity = None

        item.semantic_relevance = _blend_scores(lexical, similarity)
        item.aspect_match = topical
        item.off_topic = off_topic

    return items


def _stored_embeddings(items):
    """Embedding yang sudah tersimpan untuk baris knowledge base terkait."""
    import json

    ids = []
    for item in items or []:
        kind, _, raw = (item.source_id or "").partition(":")
        if kind == "journal" and raw.isdigit():
            ids.append(int(raw))
    if not ids:
        return {}

    vectors = {}
    try:
        JournalArticle = runtime.model("JournalArticle")

        rows = JournalArticle.objects.filter(id__in=ids).values_list("id", "embedding")
        for row_id, raw in rows:
            if not raw:
                continue
            try:
                vectors[f"journal:{row_id}"] = json.loads(raw)
            except (ValueError, TypeError):
                continue
    except Exception as exc:  # pragma: no cover
        logger.warning("[RETRIEVAL] gagal memuat embedding tersimpan: %s", exc)
    return vectors


def retrieve_candidates(query: str,
                        extra_terms: Optional[List[str]] = None,
                        limit: int = DEFAULT_CANDIDATE_LIMIT,
                        use_vector_index: bool = True) -> List[EvidenceItem]:
    """
    Kumpulkan kandidat evidence dari seluruh sumber pengetahuan yang tersedia.
    Hasil BELUM di-rerank/diseleksi — itu tugas `intelligence.evidence.selector`.
    """
    normalized = normalize_query(query)
    terms = build_search_term_groups(normalized, extra_terms=extra_terms)
    if not terms:
        return []

    # Embedding dibangun dari query yang sudah diperluas ke padanan Inggris,
    # bukan dari kalimat aslinya (lihat build_embedding_query).
    aspect_groups = build_aspect_groups(query)
    query_subjects = set(extract_conditions(query))
    query_embedding = (
        embed_query(build_embedding_query(query, extra_terms=extra_terms))
        if use_vector_index else None
    )

    candidates: List[EvidenceItem] = []
    candidates.extend(retrieve_from_journals(
        terms, limit=limit, query_embedding=query_embedding,
        aspect_groups=aspect_groups, query_subjects=query_subjects))
    candidates.extend(retrieve_from_sources(
        terms, limit=limit, aspect_groups=aspect_groups, query_subjects=query_subjects))
    if use_vector_index:
        candidates.extend(retrieve_from_vector_index(
            normalized, terms, k=10, query_embedding=query_embedding,
            aspect_groups=aspect_groups))

    unique = _dedupe(candidates)
    unique.sort(key=lambda i: i.semantic_relevance, reverse=True)

    logger.info("[RETRIEVAL] %d kandidat unik dari %d hasil mentah (terms=%s)",
                len(unique), len(candidates),
                [g["variants"][0] for g in terms[:6]])
    return unique[: limit * 2]
