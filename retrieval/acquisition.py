"""
Pelengkapan basis pengetahuan secara otomatis.

Ketika sebuah pertanyaan kesehatan tidak menemukan cukup bukti, mesin mencari
jurnal untuk topik itu ke Crossref, memverifikasi DOI-nya, menyimpannya, lalu
mengulang pengambilan. Sebelumnya lubang seperti itu hanya bisa ditutup manusia
yang menjalankan `import_journals`, dan konsumen eksternal tidak punya siapa pun
yang mengawasi.

Jurnal hasil pengambilan otomatis melewati gerbang yang sama dengan impor
manual: verifikasi DOI, ambang kemiripan makna, fokus judul, dan penyaringan
topik. Menambah bahan bacaan tidak melonggarkan syarat relevansi.

Pagarnya: hanya untuk pertanyaan yang mengandung konsep kesehatan, satu topik
sekali per `COOLDOWN_SECONDS`, kuota `HOURLY_BUDGET` per jam, batas waktu
jaringan pendek, dan setiap kegagalan ditelan sehingga permintaan pengguna tidak
ikut gagal.
"""

import hashlib
import html
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests
from django.core.cache import cache

from ..evidence import link_validator as lv
from ..lexicon import bilingual_variants
from .concepts import extract_conditions, extract_health_concepts
from .. import runtime

logger = logging.getLogger(__name__)

CROSSREF_API = "https://api.crossref.org/works"

ACCEPTED_TYPES = {"journal-article", "proceedings-article", "book-chapter", "posted-content"}
MIN_ABSTRACT_CHARS = 200

# Kata yang muncul di hampir setiap pertanyaan dan tidak menandakan topik.
_QUERY_NOISE = {
    "apakah", "adalah", "yang", "untuk", "dengan", "dari", "pada", "bisa",
    "dapat", "tidak", "itu", "ini", "akan", "harus", "kalau", "jika", "saat",
    "orang", "kita", "saya", "benar", "salah", "kenapa", "mengapa", "bagaimana",
    "does", "what", "with", "from", "that", "this", "have", "when", "where",

    # Kata penilaian. Muncul di hampir setiap klaim dan tidak menunjuk topik,
    # sehingga tanpa ini setiap pertanyaan "X berbahaya" akan dikira tidak
    # terwakili hanya karena kata "berbahaya" tidak ada di judul jurnal.
    "berbahaya", "bahaya", "aman", "sehat", "bagus", "buruk", "penting",
    "manfaat", "khasiat", "ampuh", "mujarab", "beneran", "bener",
}

# Sebuah topik tidak dicari ulang selama sehari. Bila Crossref memang tidak
# punya apa-apa untuk topik itu, mencoba lagi setiap permintaan hanya
# memperlambat jawaban tanpa mengubah hasil.
COOLDOWN_SECONDS = 24 * 60 * 60

# Kuota per jam untuk seluruh proses.
HOURLY_BUDGET = 30
BUDGET_WINDOW = 60 * 60

# Sengaja kecil. Tujuannya menutup lubang secukupnya agar pertanyaan bisa
# dijawab, bukan mengunduh seluruh literatur di tengah satu permintaan HTTP.
ROWS_PER_TOPIC = 12
FROM_YEAR = 2015
NETWORK_TIMEOUT = 12

# Berapa verifikasi DOI yang berjalan berbarengan. Cukup untuk memangkas waktu
# tunggu, masih sopan terhadap registry.
VERIFY_WORKERS = 6


def _cache_get(key, default=None):
    try:
        value = cache.get(key)
        return default if value is None else value
    except Exception:
        return default


def _cache_set(key, value, timeout):
    try:
        cache.set(key, value, timeout)
    except Exception:
        pass


def build_topic_phrase(query: str) -> str:
    """
    Ubah pertanyaan menjadi frasa pencarian berbahasa Inggris.

    Literatur di Crossref hampir seluruhnya berbahasa Inggris, sedangkan
    pertanyaan datang dalam Bahasa Indonesia. Mengirim pertanyaan apa adanya
    ("apakah covid berbahaya") menghasilkan pencarian yang buruk, jadi yang
    dikirim adalah konsep kesehatannya dalam bentuk Inggris.

    Istilah dari leksikon digabung dengan hasil terjemahan. Leksikon ditulis
    tangan sehingga selalu tertinggal; penyakit yang belum tercatat ("skabies",
    "kawasaki") hanya terjangkau lewat terjemahan.
    """
    concepts = extract_conditions(query) or extract_health_concepts(query)

    english: List[str] = []
    for concept in concepts[:4]:
        for variant in bilingual_variants(concept):
            if variant.isascii() and variant not in english:
                english.append(variant)

    # Terjemahan SELALU digabungkan, bukan hanya ketika leksikon gagal total.
    # Pengenalan sebagian justru menyesatkan: "skabies menular lewat sentuhan
    # kulit" mengenali "kulit" lalu mencari literatur kulit secara umum,
    # sementara penyakit yang sebenarnya ditanyakan tidak pernah ikut dicari.
    # Menentukan kapan pengenalan sudah "cukup" hanya bisa lewat daftar kata
    # buatan tangan, dan daftar seperti itu selalu tertinggal dari kosakata
    # yang benar-benar ditanyakan orang.
    translated = _translated_topic(query)

    parts = ([translated] if translated else []) + english[:6]
    return " ".join(parts).strip()[:250]


def _translated_topic(query: str) -> str:
    """
    Terjemahkan pertanyaan ke Bahasa Inggris untuk dijadikan frasa pencarian.

    Hasilnya disimpan di cache: pertanyaan dengan topik yang sama tidak perlu
    diterjemahkan berulang kali. Kegagalan mengembalikan string kosong, yang
    berarti pelengkapan dilewati, bukan permintaan yang ikut gagal.
    """
    normalized = " ".join(query.lower().split())
    if not normalized:
        return ""

    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    cached = _cache_get(f"acq:tr:{digest}")
    if cached is not None:
        return cached

    try:
        translate = runtime.service("translate")
        translated = (translate(query, "en") or "").strip() if translate else ""
    except Exception as exc:
        logger.warning("[ACQ] terjemahan topik gagal: %s", exc)
        translated = ""

    # Penerjemah yang mengembalikan teks aslinya berarti tidak melakukan apa
    # pun; mengirim Bahasa Indonesia ke Crossref hanya membuang permintaan.
    if not translated or translated.strip().lower() == normalized:
        translated = ""

    _cache_set(f"acq:tr:{digest}", translated, COOLDOWN_SECONDS)
    return translated[:200]


def coverage_is_thin(query: str, evidence) -> bool:
    """
    Apakah bukti yang terkumpul benar-benar membahas yang ditanyakan.

    Memicu pelengkapan hanya ketika bukti KOSONG ternyata terlalu sempit:
    pertanyaan tentang skabies bisa saja menarik lima paper penyakit kulit lain
    dan dianggap cukup, padahal tak satu pun menyebut skabies. Itu persis
    keluhan yang paling merusak kepercayaan, dan sistem harus menyadarinya
    sendiri alih-alih menunggu ada yang melaporkan.

    Diperiksa dengan cara yang tidak bergantung pada leksikon: bila tidak satu
    pun kata isi dari pertanyaan (atau padanan Inggrisnya, bila diketahui)
    muncul di judul atau ringkasan bukti mana pun, topiknya belum terwakili.
    """
    haystack = " ".join(
        f"{getattr(item, 'title', '') or ''} {getattr(item, 'snippet', '') or ''}"
        for item in (evidence or [])
    ).lower()
    if not haystack.strip():
        return True

    # Bila pertanyaan menyebut penyakit yang dikenali, itulah ukurannya:
    # cukup salah satunya (atau padanan Inggrisnya) muncul di bukti. Tanpa
    # aturan ini kata Indonesia biasa seperti "menular" atau "makanan" membuat
    # pertanyaan yang sudah terjawab lengkap tetap dinilai tidak terwakili,
    # dan setiap pertanyaan memicu pengambilan yang tidak perlu.
    conditions = extract_conditions(query) or []
    if conditions:
        wanted = set()
        for condition in conditions:
            wanted.add(condition.lower())
            for variant in bilingual_variants(condition):
                wanted.add(variant.lower())
        return not any(term in haystack for term in wanted)

    # Tidak ada penyakit yang dikenali. Kandidatnya adalah kata yang belum
    # tercatat di leksikon — persis keadaan penyakit seperti "skabies" atau
    # "kawasaki", yang tidak akan pernah terlihat bila hanya bersandar pada
    # daftar buatan tangan.
    tokens = [w for w in re.findall(r"[a-z0-9]{5,}", (query or "").lower())
              if w not in _QUERY_NOISE]
    if not tokens:
        return False

    known = {c.lower() for c in (extract_health_concepts(query) or [])}
    distinctive = [w for w in tokens
                   if not any(w in term or term in w for term in known)]

    probe = distinctive or tokens
    wanted = set(probe)
    for token in probe:
        for variant in bilingual_variants(token):
            wanted.add(variant.lower())

    return not any(term in haystack for term in wanted)


def _within_budget() -> bool:
    used = _cache_get("acq:budget", 0)
    if used >= HOURLY_BUDGET:
        logger.info("[ACQ] kuota pelengkapan per jam habis (%d)", used)
        return False
    _cache_set("acq:budget", used + 1, BUDGET_WINDOW)
    return True


def _clean_abstract(raw) -> str:
    text = str(raw or "")
    text = re.sub(r"</?jats:[^>]+>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower().startswith("abstract"):
        text = text[len("abstract"):].lstrip(": ").strip()
    return text


def _first(value) -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def _authors(authors) -> str:
    if not isinstance(authors, list):
        return ""
    names = []
    for author in authors[:12]:
        given = (author.get("given") or "").strip()
        family = (author.get("family") or "").strip()
        full = f"{family}, {given}".strip(", ") if family else given
        if full:
            names.append(full)
    return "; ".join(names)


def _issued_date(issued):
    import datetime

    try:
        parts = (issued or {}).get("date-parts") or [[]]
        values = parts[0]
        if not values:
            return None
        year = int(values[0])
        month = int(values[1]) if len(values) > 1 else 1
        day = int(values[2]) if len(values) > 2 else 1
        return datetime.date(year, month, day)
    except Exception:
        return None


def search_crossref(topic: str, rows: int = ROWS_PER_TOPIC,
                    from_year: int = FROM_YEAR,
                    mailto: str = "") -> List[Dict[str, Any]]:
    params = {
        "query.bibliographic": topic,
        "rows": rows,
        "filter": f"from-pub-date:{from_year}-01-01,has-abstract:true",
        "select": "DOI,title,abstract,author,container-title,publisher,issued,type,URL,subject",
        "sort": "relevance",
    }
    if mailto:
        params["mailto"] = mailto

    response = requests.get(
        CROSSREF_API, params=params, timeout=NETWORK_TIMEOUT,
        headers={"User-Agent": f"Healthify/1.0 (mailto:{mailto})"},
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("items", []) or []


def build_record(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ubah satu hasil Crossref menjadi baris JournalArticle, atau None bila tidak
    memenuhi syarat. DOI diverifikasi ke registry di sini, sehingga tidak ada
    DOI tak terdaftar yang bisa masuk lewat jalur mana pun.
    """
    if item.get("type") not in ACCEPTED_TYPES:
        return None

    doi = lv.normalize_doi(item.get("DOI"))
    title = _first(item.get("title"))
    abstract = _clean_abstract(item.get("abstract"))

    if not doi or not title or len(abstract) < MIN_ABSTRACT_CHARS:
        return None
    if lv.resolve_doi(doi) != lv.STATUS_VERIFIED:
        return None

    return {
        "title": title[:1000],
        "abstract": abstract,
        "authors": _authors(item.get("author")),
        "doi": doi,
        "url": lv.doi_to_url(doi),
        "publisher": (item.get("publisher") or "")[:500],
        "journal_name": (_first(item.get("container-title")) or "")[:500],
        "published_date": _issued_date(item.get("issued")),
        "source_portal": "other",
        "keywords": ", ".join((item.get("subject") or [])[:8]),
    }


def _safe_record(item):
    """`build_record` yang tidak pernah melempar, supaya satu artikel rusak
    tidak menggagalkan seluruh pengambilan."""
    try:
        return build_record(item)
    except Exception as exc:
        logger.warning("[ACQ] artikel dilewati: %s", exc)
        return None


def ensure_coverage(query: str, health_checked: bool = False) -> int:
    """
    Lengkapi basis pengetahuan untuk topik `query`, kembalikan jumlah jurnal
    baru yang tersimpan.

    Pengambilan yang sesekali tidak perlu dibiarkan; yang dijaga batasnya, dan
    akibat terburuknya hanya bertambahnya jurnal ber-DOI sah untuk topik yang
    memang ditanyakan.

    `health_checked` diisi pemanggil yang sudah memastikan pertanyaannya soal
    kesehatan. Bila tidak, modul ini memeriksanya sendiri agar teks sembarang
    dari endpoint publik tidak bisa mengendalikan permintaan ke Crossref.

    Aman dipanggil di tengah permintaan: kegagalan ditelan dan mengembalikan 0.
    """
    import os

    from ..lexicon import find_aspects

    JournalArticle = runtime.model("JournalArticle")

    if not health_checked:
        recognized = extract_health_concepts(query) or find_aspects(query)
        if not recognized:
            return 0

    topic = build_topic_phrase(query)
    if not topic:
        return 0


    # Topik mengandung spasi dan tanda baca; sebagian backend cache menolak
    # kunci seperti itu. Hash memberi kunci yang selalu sah.
    digest = hashlib.sha256(topic.lower().encode("utf-8")).hexdigest()[:32]
    marker = f"acq:topic:{digest}"
    if _cache_get(marker):
        return 0
    _cache_set(marker, 1, COOLDOWN_SECONDS)

    if not _within_budget():
        return 0

    try:
        items = search_crossref(topic, mailto=os.getenv("CROSSREF_MAILTO", ""))
    except Exception as exc:
        logger.warning("[ACQ] pencarian gagal untuk %r: %s", topic, exc)
        return 0

    # Saring yang sudah ada lebih dulu, dengan SATU query, bukan satu query per
    # artikel.
    dois = {}
    for item in items:
        doi = lv.normalize_doi(item.get("DOI"))
        if doi:
            dois.setdefault(doi, item)
    if not dois:
        return 0

    existing = set(JournalArticle.objects.filter(doi__in=list(dois))
                   .values_list("doi", flat=True))
    fresh = [item for doi, item in dois.items() if doi not in existing]
    if not fresh:
        return 0

    # Verifikasi DOI berjalan berbarengan. Berurutan, dua belas artikel berarti
    # dua belas perjalanan jaringan yang saling menunggu; permintaan pengguna
    # ikut menunggu seluruhnya.
    records = []
    with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as pool:
        for record in pool.map(_safe_record, fresh):
            if record:
                records.append(record)

    created = 0
    for record in records:
        try:
            JournalArticle.objects.create(**record)
            created += 1
        except Exception as exc:
            logger.warning("[ACQ] gagal menyimpan satu artikel: %s", exc)

    if created:
        logger.info("[ACQ] %d jurnal baru untuk topik %r", created, topic)
        _embed_new_articles()
    return created


_embedding_lock = threading.Lock()


def _embed_new_articles() -> None:
    """
    Buat embedding untuk jurnal yang belum punya, di luar permintaan.

    Satu embedding memakan lebih dari satu detik. Mengerjakan dua puluh empat
    di dalam permintaan menambahkan hampir setengah menit ke waktu tunggu
    pengguna, dan itulah penyebab terbesar lambatnya jawaban pertama untuk
    sebuah topik baru.

    Menundanya tidak membuat jurnal baru hilang dari jangkauan: tanpa embedding
    ia tetap ditemukan lewat pencocokan kata kunci, yang memang jalur yang
    dipakai percobaan ulang (istilah Inggris hasil terjemahan). Embedding
    hanya menambah ketajaman, dan menyusul beberapa detik kemudian.

    Hanya satu utas yang berjalan pada satu waktu, supaya lonjakan pertanyaan
    baru tidak berubah menjadi lonjakan pemanggilan penyedia embedding.
    """
    if not _embedding_lock.acquire(blocking=False):
        return

    def worker():
        try:
            from django.db import connection

            JournalArticle = runtime.model("JournalArticle")
            embed_article = runtime.service("embed_article")
            if embed_article is None:
                return

            try:
                pending = list(JournalArticle.objects.filter(is_embedded=False)[:64])
                for article in pending:
                    try:
                        embed_article(article)
                    except Exception:
                        continue
                if pending:
                    logger.info("[ACQ] %d embedding menyusul di latar belakang", len(pending))
            finally:
                # Koneksi milik utas ini tidak dipakai siapa pun lagi.
                connection.close()
        finally:
            _embedding_lock.release()

    threading.Thread(target=worker, name="healthify-embed", daemon=True).start()
