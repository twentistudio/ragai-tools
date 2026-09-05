"""
Titik sambung ke aplikasi induk.

Engine hanya menyebut apa yang dibutuhkannya; aplikasi induk mendaftarkan
pemenuhnya saat start. Tanpa ini engine harus mengimpor aplikasi induk, dan
tidak bisa dipakai di tempat lain.

    runtime.configure(models={...}, services={...}, config={...})
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
_config: Dict[str, Any] = {}


def configure(models: Optional[Dict[str, Any]] = None,
              services: Optional[Dict[str, Callable]] = None,
              config: Optional[Dict[str, Any]] = None) -> None:
    """Daftarkan model, layanan, dan konfigurasi milik aplikasi induk."""
    if models:
        _models.update(models)
    if services:
        _services.update(services)
    if config:
        _config.update(config)

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


def config_value(name: str) -> Optional[Any]:
    """Nilai konfigurasi yang didaftarkan induk. Dibaca lewat `ragai.config`."""
    return _config.get(name)


def is_configured() -> bool:
    """Apakah seluruh model yang dibutuhkan sudah terdaftar."""
    return all(name in _models for name in REQUIRED_MODELS)


def reset() -> None:
    """Kosongkan pendaftaran. Dipakai pengujian yang menukar implementasi."""
    _models.clear()
    _services.clear()
    _config.clear()
