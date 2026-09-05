"""
Query normalization & health concept extraction (§10, tahap 1-2).

Dipisah dari retriever supaya bisa diuji sendiri dan dipakai ulang oleh
claim engine maupun consultation engine.
"""

import re
from typing import List

from ..lexicon import (
    ALL_SYMPTOM_VARIANTS,
    canonical_condition,
    CONDITION_TRANSLATIONS,
    SYMPTOM_TERMS,
    aspect_variants,
    find_aspects,
    find_translated_concepts,
    CONDITION_TERMS,
    GENERAL_HEALTH_TERMS,
    MEDICATION_TERMS,
    SYMPTOM_LOOKUP,
    bilingual_variants,
    find_symptom_variants,
)

_STOPWORDS = {
    "yang", "dan", "atau", "dengan", "untuk", "pada", "dari", "ini", "itu",
    "adalah", "akan", "dapat", "bisa", "tidak", "juga", "oleh", "dalam",
    "sebagai", "karena", "jika", "agar", "lebih", "sudah", "telah", "masih",
    "saya", "aku", "kami", "kita", "anda", "mereka", "apakah", "apa",
    "bagaimana", "kenapa", "mengapa", "kapan", "dimana", "berapa", "sih",
    "the", "and", "or", "with", "for", "from", "this", "that", "are", "was",
    "have", "has", "can", "does", "did", "what", "how", "why", "when", "where",
    "is", "it", "my", "i", "you", "your", "a", "an", "of", "to", "in", "on",

    # "air" ditulis sama persis dalam Bahasa Indonesia (zat cair) dan Bahasa
    # Inggris (udara). Sebagai istilah pencarian tunggal ia menarik paper
    # polusi udara untuk pertanyaan tentang minum air. Frasa yang memuatnya
    # tetap terpakai lewat leksikon ("air putih" -> drinking water), yang
    # justru merupakan bentuk yang berguna.
    "air",
}

_WORD_RE = re.compile(r"[a-zA-Zà-ÿ0-9]+")

# Bobot token yang bukan konsep kesehatan (kata pengisi kalimat).
GENERIC_TOKEN_WEIGHT = 0.2


def normalize_query(text: str) -> str:
    """Rapikan query: lowercase, buang tanda baca berlebih, rapatkan spasi."""
    if not text:
        return ""
    normalized = str(text).lower().strip()
    normalized = re.sub(r"[^\w\s\-/%.]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize(text: str) -> List[str]:
    return [
        w.lower() for w in _WORD_RE.findall(text or "")
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]


def _scan_conditions(low: str):
    """
    Nama penyakit yang muncul di teks, tanpa rentang yang saling menimpa.

    Frasa terpanjang diperiksa lebih dulu dan rentang teks yang sudah diklaim
    tidak dipakai ulang. "gula darah tinggi" memuat "darah tinggi" di dalamnya;
    tanpa penekan ini satu pertanyaan tentang gula darah juga tercatat sebagai
    hipertensi, sehingga paper tekanan darah lolos gerbang fokus judul seolah
    membahas hal yang ditanyakan.

    Aturan yang sama sudah dipakai `find_symptom_variants` untuk gejala.
    """
    claimed = []
    for condition in sorted(CONDITION_TERMS, key=len, reverse=True):
        start = low.find(condition)
        while start != -1:
            end = start + len(condition)
            if not any(start < c_end and end > c_start for c_start, c_end in claimed):
                claimed.append((start, end))
                yield condition
                break
            start = low.find(condition, start + 1)


def extract_health_concepts(text: str) -> List[str]:
    """
    Ambil konsep kesehatan (gejala kanonik, kondisi, obat, istilah medis)
    yang benar-benar muncul di teks. Urutan: paling spesifik dulu.
    """
    low = normalize_query(text)
    concepts: List[str] = []

    def add(value: str):
        value = (value or "").strip().lower()
        if value and value not in concepts:
            concepts.append(value)

    for variant in find_symptom_variants(low):
        add(SYMPTOM_LOOKUP[variant])

    for condition in _scan_conditions(low):
        add(condition)

    # Kenali juga istilah berbahasa Inggris dan petakan ke bentuk kanoniknya,
    # sehingga judul jurnal Inggris menghasilkan konsep yang sama dengan
    # pertanyaan berbahasa Indonesia.
    for canonical in find_translated_concepts(low):
        add(canonical)

    for med in MEDICATION_TERMS:
        if len(med) > 5 and med in low:
            add(med)

    for term in GENERAL_HEALTH_TERMS:
        if len(term) > 4 and term in low:
            add(term)

    return concepts


def build_search_terms(query: str, extra_terms: List[str] = None) -> List[str]:
    """Gabungan konsep kesehatan + token bermakna, untuk pencocokan leksikal."""
    terms: List[str] = []
    for concept in extract_health_concepts(query):
        if concept not in terms:
            terms.append(concept)
    for term in (extra_terms or []):
        low = (term or "").strip().lower()
        if low and low not in terms:
            terms.append(low)
    for token in tokenize(query):
        if token not in terms:
            terms.append(token)
    return terms


def build_search_term_groups(query: str, extra_terms: List[str] = None,
                             max_generic_tokens: int = 5) -> List[dict]:
    """
    Seperti `build_search_terms`, tapi setiap istilah dikembangkan menjadi
    **kelompok varian bilingual** (ID + EN) dan diberi **bobot**.

    Dua alasan:

    1. Knowledge base Healthify banyak berbahasa Inggris sementara pengguna
       bertanya dalam Bahasa Indonesia. Tanpa pengembangan varian,
       "demam berdarah" tidak akan pernah cocok dengan abstrak "dengue fever".

    2. Kalimat percakapan penuh token generik ("tinggi", "tiga", "hari").
       Bila semuanya dihitung setara dengan konsep medis, skor relevansi
       terdilusi dan dokumen yang benar ikut tenggelam. Konsep kesehatan
       karena itu diberi bobot penuh, token sisanya bobot kecil.

    Returns: list of {"variants": [...], "weight": float, "is_concept": bool}
    """
    groups: List[dict] = []
    seen = set()

    def add(term: str, is_concept: bool):
        term = (term or "").strip().lower()
        if not term or term in seen:
            return
        seen.add(term)
        variants = bilingual_variants(term) or [term]
        weight = (1.0 + min(len(term.split()), 3) * 0.5) if is_concept else GENERIC_TOKEN_WEIGHT
        groups.append({"variants": variants, "weight": weight, "is_concept": is_concept})

    for concept in extract_health_concepts(query):
        add(concept, True)
    for term in (extra_terms or []):
        add(term, True)

    generic = 0
    for token in tokenize(query):
        if token in seen:
            continue
        add(token, False)
        generic += 1
        if generic >= max_generic_tokens:
            break

    return groups


def build_aspect_groups(query: str) -> List[List[str]]:
    """
    Aspek yang ditanyakan, masing-masing beserta seluruh variannya.

    Dipakai untuk menilai apakah sebuah dokumen benar-benar membahas hal yang
    ditanyakan, bukan sekadar menyebut penyakit yang sama.
    """
    groups: List[List[str]] = []
    for aspect in find_aspects(query):
        variants = aspect_variants(aspect)
        if variants:
            groups.append(variants)
    return groups


def build_embedding_query(query: str, extra_terms: List[str] = None) -> str:
    """
    Susun teks yang akan di-embed untuk pencarian semantik.

    Kenapa bukan kalimat aslinya
    ----------------------------
    Basis pengetahuan sebagian besar berbahasa Inggris, sementara pengguna
    bertanya dalam Bahasa Indonesia. Meng-embed kalimat Indonesia mentah
    menghasilkan kemiripan yang rendah dan nyaris seragam: pertanyaan yang
    memang ada jawabannya dan yang tidak ada sama-sama berada di sekitar 0,33,
    sehingga sinyal semantik tidak dapat membedakan keduanya.

    Dengan menyertakan padanan Inggris dari setiap konsep dan aspek, kemiripan
    untuk pertanyaan yang terjawab naik tajam sementara yang tidak terjawab
    tetap rendah, sehingga ambang batas menjadi bermakna.
    """
    # Seluruh varian disertakan, bukan hanya beberapa yang pertama: daftar
    # varian tersusun Indonesia dahulu, sehingga memotongnya justru membuang
    # padanan Bahasa Inggris yang menjadi tujuan perluasan ini.
    parts: List[str] = [normalize_query(query)]

    for concept in extract_health_concepts(query):
        parts.extend(bilingual_variants(concept))

    for term in (extra_terms or []):
        parts.extend(bilingual_variants(term))

    for aspect in find_aspects(query):
        parts.extend(aspect_variants(aspect))

    seen = set()
    unique = []
    for part in parts:
        part = (part or "").strip().lower()
        if part and part not in seen:
            seen.add(part)
            unique.append(part)
    return " ".join(unique)


def extract_conditions(text: str) -> List[str]:
    """
    Hanya PENYAKIT dan GEJALA yang disebut, tanpa kosakata kesehatan umum.

    Dipakai untuk menilai apakah sebuah dokumen membahas penyakit yang
    ditanyakan. Menyertakan kosakata umum ("antibiotik", "pengobatan",
    "infeksi") menggelembungkan penyebut sehingga judul deskriptif yang panjang
    tampak melenceng padahal seluruh sub-topiknya milik penyakit yang sama.
    """
    low = normalize_query(text)
    found: List[str] = []

    def add(value: str):
        value = (value or "").strip().lower()
        if value and value not in found:
            found.append(value)

    for variant in find_symptom_variants(low):
        add(SYMPTOM_LOOKUP[variant])

    for condition in _scan_conditions(low):
        add(condition)

    for canonical in find_translated_concepts(low):
        if canonical in CONDITION_TRANSLATIONS or canonical in SYMPTOM_TERMS:
            add(canonical)

    # Satukan sebutan yang berbeda untuk penyakit yang sama, supaya pertanyaan
    # dan judul dokumen dibandingkan dalam istilah yang sama.
    return [canonical_condition(c) for c in found]
