"""
Titik sambung antara engine dan aplikasi yang memakainya.

Engine ini berdiri sendiri sebagai produk (ragai) tetapi tetap membutuhkan
beberapa hal dari aplikasi induknya: tempat menyimpan jurnal dan percakapan,
cara membuat embedding, dan cara menerjemahkan teks. Sebelumnya kebutuhan itu
dipenuhi dengan mengimpor langsung dari aplikasi Healthify, sehingga engine
tidak bisa dipakai di tempat lain tanpa membawa serta seluruh produknya.

Sekarang engine hanya menyebut APA yang dibutuhkannya; aplikasi induk yang
mendaftarkan pemenuhnya saat start. Healthify melakukannya di `api/apps.py`.

Contoh pemasangan dari aplikasi lain:

    from ragai import runtime

    runtime.configure(
        models={"JournalArticle": ..., "ConversationSession": ...},
        services={"translate": ..., "embed_article": ...},
    )
"""

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Model yang dibutuhkan engine, dengan peran masing-masing. Nama kelasnya bebas
# di sisi aplikasi induk; yang mengikat hanya kunci di bawah ini.
REQUIRED_MODELS = (
    "JournalArticle",        # basis pengetahuan jurnal
    "Source",                # sumber yang pernah terkait dengan klaim
    "ClaimSource",           # kutipan yang menghubungkan klaim dan sumber
    "ConversationSession",   # sesi percakapan multi-giliran
    "ConversationMessage",   # satu giliran dalam sesi
    "ConsultationSummary",   # ringkasan hasil konsultasi
)

# Layanan opsional. Tanpa ini engine tetap berjalan, hanya kehilangan sebagian
# kemampuan: tanpa `translate` kosakata di luar leksikon tidak terjangkau, dan
# tanpa `embed_article` jurnal baru hanya ditemukan lewat kata kunci.
OPTIONAL_SERVICES = ("translate", "embed_article", "training_scripts_dir",
                     "training_modules_available")

_models: Dict[str, Any] = {}
_services: Dict[str, Callable] = {}


def configure(models: Optional[Dict[str, Any]] = None,
              services: Optional[Dict[str, Callable]] = None) -> None:
    """Daftarkan model dan layanan milik aplikasi induk."""
    if models:
        _models.update(models)
    if services:
        _services.update(services)

    missing = [name for name in REQUIRED_MODELS if name not in _models]
    if missing:
        logger.warning("[RAGAI] model belum terdaftar: %s", ", ".join(missing))


def model(name: str):
    """
    Kelas model untuk peran `name`.

    Melempar bila belum terdaftar: engine yang berjalan tanpa tempat menyimpan
    apa pun akan gagal dengan cara yang membingungkan jauh dari sumbernya.
    """
    try:
        return _models[name]
    except KeyError:
        raise RuntimeError(
            f"Model '{name}' belum didaftarkan ke ragai. "
            f"Panggil ragai.runtime.configure(models={{...}}) saat aplikasi start."
        ) from None


def service(name: str) -> Optional[Callable]:
    """Layanan opsional, atau None bila aplikasi induk tidak menyediakannya."""
    return _services.get(name)


def is_configured() -> bool:
    """Apakah seluruh model yang dibutuhkan sudah terdaftar."""
    return all(name in _models for name in REQUIRED_MODELS)


def reset() -> None:
    """Kosongkan pendaftaran. Dipakai pengujian yang menukar implementasi."""
    _models.clear()
    _services.clear()
