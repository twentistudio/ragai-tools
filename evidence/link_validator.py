"""
Verifikasi DOI dan URL.

DOI diperiksa ke Handle System dan Crossref sebelum dijadikan tautan, dan
judul diambil dari registry. Tanpa ini pembaca menerima tautan yang berujung
404 atau membuka paper yang berbeda dari judulnya.
"""

import logging
import re
from typing import Optional, Tuple
from .. import config

logger = logging.getLogger(__name__)

# Status hasil validasi
STATUS_VERIFIED = "verified"        # dipastikan ada
STATUS_UNRESOLVABLE = "unresolvable"  # dipastikan TIDAK ada / mati
STATUS_UNKNOWN = "unknown"          # tidak bisa dipastikan (jaringan/timeout)
STATUS_MALFORMED = "malformed"      # bukan format DOI yang sah
STATUS_SKIPPED = "skipped"          # tidak ada yang perlu dicek

# Pola DOI resmi (ANSI/NISO Z39.84)
_DOI_RE = re.compile(r"^10\.\d{4,9}/[-._;()/:a-z0-9<>\[\]+]+$", re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(
    r"^\s*(?:urn:)?(?:doi:|info:doi/|https?://(?:dx\.)?doi\.org/|https?://doi\.org/)\s*",
    re.IGNORECASE,
)

_DEFAULT_TIMEOUT = 5.0
_CACHE_TTL_VERIFIED = 60 * 60 * 24 * 30   # 30 hari
_CACHE_TTL_NEGATIVE = 60 * 60 * 24        # 1 hari (siapa tahu baru terdaftar)
_CACHE_TTL_UNKNOWN = 60 * 5               # 5 menit


def _get_cache():
    try:
        from django.core.cache import cache
        return cache
    except Exception:  # pragma: no cover - Django selalu ada di runtime app
        return None


def _cache_get(key, default=None):
    """Baca cache dengan aman.

    Cache berbasis database bisa belum siap (tabel belum dibuat) atau sedang
    bermasalah. Kegagalan cache tidak boleh mematikan permintaan; paling buruk
    kita kehilangan sedikit kecepatan.
    """
    cache = _get_cache()
    if cache is None:
        return default
    try:
        value = cache.get(key)
    except Exception as exc:
        logger.debug("[LINK] pembacaan cache gagal (%s): %s", key, exc)
        return default
    return default if value is None else value


def _cache_set(key, value, timeout):
    cache = _get_cache()
    if cache is None:
        return
    try:
        cache.set(key, value, timeout=timeout)
    except Exception as exc:
        logger.debug("[LINK] penulisan cache gagal (%s): %s", key, exc)


def normalize_doi(raw: Optional[str]) -> str:
    """Bersihkan DOI dari prefix/URL dan karakter sampah di ujung."""
    if not raw:
        return ""
    doi = str(raw).strip()
    doi = _DOI_PREFIX_RE.sub("", doi)
    doi = doi.strip().strip('.,;"\'()<>[]')
    # Buang query string / fragment yang kadang ikut ter-copy
    doi = re.split(r"[?#]", doi)[0]
    return doi.strip()


def looks_like_doi(doi: str) -> bool:
    """Cek cepat format DOI tanpa akses jaringan."""
    return bool(doi) and bool(_DOI_RE.match(doi.strip()))


def doi_to_url(doi: str) -> str:
    doi = normalize_doi(doi)
    return f"https://doi.org/{doi}" if doi else ""


def _http_get(url: str, timeout: float, headers=None):
    import requests
    return requests.get(url, timeout=timeout, headers=headers or {}, allow_redirects=True)


def fetch_doi_metadata(doi: str, timeout: float = _DEFAULT_TIMEOUT,
                       use_cache: bool = True) -> Optional[dict]:
    """
    Ambil metadata resmi sebuah DOI dari Crossref.
    Memastikan sebuah DOI *terdaftar* ternyata tidak cukup. LLM dapat
    mengarang judul yang meyakinkan lalu memasangkannya dengan DOI yang
    kebetulan benar-benar ada, tetapi milik paper lain. Hasilnya: judul di
    layar tidak sesuai dengan halaman yang terbuka saat diklik.

    Satu-satunya sumber kebenaran untuk judul sebuah DOI adalah registry.

    Returns dict {title, publisher, container, year, authors, type} atau None
    bila metadata tidak tersedia (mis. DOI terdaftar di agensi non-Crossref).
    """
    doi = normalize_doi(doi)
    if not doi or not looks_like_doi(doi):
        return None
    if config.get_bool("EVIDENCE_LINK_CHECK_ENABLED", True) is False:
        return None

    cache_key = f"doi_meta:v1:{doi.lower()}"
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached or None

    metadata = None
    try:
        import os

        mailto = config.get("CROSSREF_MAILTO", "")
        resp = _http_get(
            f"https://api.crossref.org/works/{doi}",
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": f"Healthify/1.0 (mailto:{mailto})",
            },
        )
        if resp.status_code == 200:
            message = resp.json().get("message") or {}
            titles = message.get("title") or []
            issued = ((message.get("issued") or {}).get("date-parts") or [[]])[0]
            authors = []
            for author in (message.get("author") or [])[:12]:
                family = (author.get("family") or "").strip()
                given = (author.get("given") or "").strip()
                full = f"{family}, {given}".strip(", ") if family else given
                if full:
                    authors.append(full)
            metadata = {
                "title": (titles[0].strip() if titles else ""),
                "publisher": (message.get("publisher") or "").strip(),
                "container": ((message.get("container-title") or [""])[0] or "").strip(),
                "year": (int(issued[0]) if issued and str(issued[0]).isdigit() else None),
                "authors": "; ".join(authors),
                "type": message.get("type") or "",
            }
    except Exception as exc:
        logger.debug("[LINK] metadata Crossref gagal untuk %s: %s", doi, exc)

    _cache_set(cache_key, metadata or {},
               _CACHE_TTL_VERIFIED if metadata else _CACHE_TTL_UNKNOWN)
    return metadata


def titles_match(stored: str, registered: str) -> bool:
    """
    Apakah judul tersimpan merujuk karya yang sama dengan judul di registry?

    Perbandingan longgar: huruf kecil, tanpa tanda baca, mengabaikan subjudul
    setelah titik dua. Perbedaan kecil (tanda hubung, tanda kutip, spasi) tidak
    dianggap ketidakcocokan, tetapi judul yang benar-benar berbeda dianggap iya.
    """
    def norm(value: str) -> set:
        value = re.sub(r"<[^>]+>", " ", (value or "").lower())
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        return {w for w in value.split() if len(w) > 2}

    a, b = norm(stored), norm(registered)
    if not a or not b:
        return True  # tidak cukup data untuk menuduh

    overlap = len(a & b) / float(min(len(a), len(b)))
    return overlap >= 0.6


def resolve_doi(doi: str, timeout: float = _DEFAULT_TIMEOUT, use_cache: bool = True) -> str:
    """
    Cek apakah sebuah DOI benar-benar terdaftar.

    Returns: STATUS_VERIFIED | STATUS_UNRESOLVABLE | STATUS_UNKNOWN | STATUS_MALFORMED
    """
    doi = normalize_doi(doi)
    if not doi:
        return STATUS_SKIPPED
    if not looks_like_doi(doi):
        logger.info("[LINK] DOI malformed, ditolak: %r", doi[:120])
        return STATUS_MALFORMED

    if config.get_bool("EVIDENCE_LINK_CHECK_ENABLED", True) is False:
        return STATUS_UNKNOWN

    cache_key = f"doi_resolve:v1:{doi.lower()}"
    if use_cache:
        cached = _cache_get(cache_key)
        if cached in (STATUS_VERIFIED, STATUS_UNRESOLVABLE):
            return cached

    status = _resolve_doi_uncached(doi, timeout)
    _cache_set(cache_key, status, {
        STATUS_VERIFIED: _CACHE_TTL_VERIFIED,
        STATUS_UNRESOLVABLE: _CACHE_TTL_NEGATIVE,
    }.get(status, _CACHE_TTL_UNKNOWN))
    return status


def _resolve_doi_uncached(doi: str, timeout: float) -> str:
    """Cek DOI ke Handle System, fallback ke Crossref."""
    # 1) DOI Handle System, sumber kebenaran resmi.
    #    responseCode 1 = handle ada, 100 = handle tidak ditemukan.
    try:
        resp = _http_get(
            f"https://doi.org/api/handles/{doi}",
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 200:
            data = resp.json()
            code = data.get("responseCode")
            if code == 1:
                return STATUS_VERIFIED
            if code in (100, 200):  # 100 = handle not found, 200 = values not found
                logger.info("[LINK] DOI tidak terdaftar di Handle System: %s", doi)
                return STATUS_UNRESOLVABLE
        elif resp.status_code == 404:
            logger.info("[LINK] DOI 404 di Handle System: %s", doi)
            return STATUS_UNRESOLVABLE
    except Exception as exc:
        logger.debug("[LINK] Handle System gagal untuk %s: %s", doi, exc)

    # 2) Fallback: Crossref agency endpoint.
    try:
        resp = _http_get(
            f"https://api.crossref.org/works/{doi}/agency",
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 200:
            return STATUS_VERIFIED
        if resp.status_code == 404:
            return STATUS_UNRESOLVABLE
    except Exception as exc:
        logger.debug("[LINK] Crossref gagal untuk %s: %s", doi, exc)

    return STATUS_UNKNOWN


def check_url(url: str, timeout: float = _DEFAULT_TIMEOUT, use_cache: bool = True) -> Tuple[str, str]:
    """
    Cek URL non-DOI.

    Returns: (status, final_url). `final_url` kosong bila URL tidak layak dipakai.
    """
    url = (url or "").strip()
    if not url:
        return STATUS_SKIPPED, ""
    if not url.lower().startswith(("http://", "https://")):
        return STATUS_MALFORMED, ""

    if config.get_bool("EVIDENCE_LINK_CHECK_ENABLED", True) is False:
        return STATUS_UNKNOWN, url

    cache_key = f"url_check:v1:{url}"
    if use_cache:
        cached = _cache_get(cache_key)
        if isinstance(cached, (list, tuple)) and len(cached) == 2:
            return cached[0], cached[1]

    status, final_url = STATUS_UNKNOWN, url
    try:
        import requests
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        code = resp.status_code
        # Sebagian server menolak HEAD; coba GET ringan.
        if code in (403, 405, 501):
            resp = requests.get(url, allow_redirects=True, timeout=timeout, stream=True)
            code = resp.status_code
        if code in (404, 410) or code >= 500:
            logger.info("[LINK] URL tidak dapat dijangkau (%s): %s", code, url)
            status, final_url = STATUS_UNRESOLVABLE, ""
        elif 200 <= code < 400:
            status, final_url = STATUS_VERIFIED, (getattr(resp, "url", "") or url)
    except Exception as exc:
        logger.debug("[LINK] cek URL gagal untuk %s: %s", url, exc)
        status, final_url = STATUS_UNKNOWN, url

    _cache_set(cache_key, (status, final_url), {
        STATUS_VERIFIED: _CACHE_TTL_VERIFIED,
        STATUS_UNRESOLVABLE: _CACHE_TTL_NEGATIVE,
    }.get(status, _CACHE_TTL_UNKNOWN))
    return status, final_url


def validate_reference(doi: Optional[str], url: Optional[str] = "",
                       timeout: float = _DEFAULT_TIMEOUT,
                       trust_on_unknown: bool = False):
    """
    Validasi pasangan (doi, url) dan kembalikan bentuk yang AMAN dipublikasikan.

    Args:
        trust_on_unknown: hanya untuk sumber yang berasal dari knowledge base
            Healthify sendiri (dikurasi admin). Bila True dan status tidak bisa
            dipastikan karena masalah jaringan, link tetap diberikan.
            Untuk sumber yang berasal dari LLM, wajib False.

    Returns dict:
        {
          "doi": str,            # "" bila DOI tidak sah / tidak ada
          "url": str,            # "" bila tidak ada link yang bisa dipertanggungjawabkan
          "doi_verified": bool,
          "link_status": str,
        }

    Prinsip: lebih baik TANPA link daripada link mati.
    """
    doi_norm = normalize_doi(doi)
    url_in = (url or "").strip()

    if doi_norm:
        status = resolve_doi(doi_norm, timeout=timeout)
        if status == STATUS_VERIFIED:
            return {
                "doi": doi_norm,
                "url": doi_to_url(doi_norm),
                "doi_verified": True,
                "link_status": STATUS_VERIFIED,
            }
        if status in (STATUS_UNRESOLVABLE, STATUS_MALFORMED):
            # DOI dipastikan/terindikasi palsu -> buang DOI-nya sama sekali,
            # lalu coba selamatkan URL asli (kalau ada dan bukan turunan DOI).
            logger.warning("[LINK] Membuang DOI tidak valid (%s): %r", status, str(doi)[:120])
            if url_in and "doi.org/" not in url_in.lower():
                u_status, final_url = check_url(url_in, timeout=timeout)
                if u_status == STATUS_VERIFIED:
                    return {"doi": "", "url": final_url, "doi_verified": False,
                            "link_status": STATUS_VERIFIED}
            return {"doi": "", "url": "", "doi_verified": False, "link_status": status}

        # STATUS_UNKNOWN: format DOI sah tapi tidak bisa dikonfirmasi sekarang.
        # Sumber kurasi internal boleh tetap memberi link; sumber lain tidak.
        return {
            "doi": doi_norm,
            "url": doi_to_url(doi_norm) if trust_on_unknown else "",
            "doi_verified": False,
            "link_status": STATUS_UNKNOWN,
        }

    if url_in:
        u_status, final_url = check_url(url_in, timeout=timeout)
        if u_status == STATUS_VERIFIED:
            return {"doi": "", "url": final_url, "doi_verified": False, "link_status": STATUS_VERIFIED}
        if u_status == STATUS_UNKNOWN:
            # Tidak terkonfirmasi: tautan ditahan, kecuali sumber internal.
            return {
                "doi": "",
                "url": final_url if trust_on_unknown else "",
                "doi_verified": False,
                "link_status": STATUS_UNKNOWN,
            }
        return {"doi": "", "url": "", "doi_verified": False, "link_status": u_status}

    return {"doi": "", "url": "", "doi_verified": False, "link_status": STATUS_SKIPPED}
