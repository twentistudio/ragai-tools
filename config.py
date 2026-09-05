"""
Konfigurasi engine.

Urutan pembacaan: nilai yang didaftarkan aplikasi induk, setelan Django bila
induknya Django, variabel lingkungan, lalu nilai bawaan. Daftar variabelnya ada
di `.env.example`.
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FALSE = ("0", "false", "no", "off", "")


def _from_host(name: str) -> Optional[Any]:
    """Nilai yang didaftarkan aplikasi induk, bila ada."""
    from . import runtime

    return runtime.config_value(name)


# Penanda "setelan tidak ada", dibedakan dari setelan yang ada tetapi kosong.
_ABSENT = object()


def _from_django(name: str) -> Any:
    try:
        from django.conf import settings

        return getattr(settings, name, _ABSENT)
    except Exception:  # pragma: no cover - induk bukan Django
        return _ABSENT


def get(name: str, default: Any = None) -> Any:
    """
    Nilai konfigurasi `name`.

    Setelan yang dinyatakan aplikasi induk dihormati apa adanya, termasuk bila
    isinya kosong: mengosongkan `LLM_PROVIDER` adalah cara menyatakan "tanpa
    preferensi", bukan lupa mengisi.

    Variabel lingkungan diperlakukan sebaliknya. Nama yang dideklarasikan tanpa
    nilai adalah keadaan lazim di berkas `.env`, dan tidak boleh menutupi nilai
    bawaan.
    """
    value = _from_host(name)
    if value is not None:
        return value

    value = _from_django(name)
    if value is not _ABSENT and value is not None:
        return value

    value = os.getenv(name)
    return default if value in (None, "") else value


def get_bool(name: str, default: bool = True) -> bool:
    """Nilai boolean; "0", "false", "no", dan "off" dibaca sebagai salah."""
    value = get(name)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSE


def get_int(name: str, default: int) -> int:
    """Nilai bilangan bulat, kembali ke bawaan bila tidak dapat dibaca."""
    value = get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("[RAGAI] %s bukan bilangan (%r); memakai %s", name, value, default)
        return default


def get_dict(name: str, default: Optional[dict] = None) -> dict:
    """Nilai berbentuk dict, mis. bobot penilaian bukti."""
    value = get(name)
    return dict(value) if isinstance(value, dict) else dict(default or {})
