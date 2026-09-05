"""
Klien LLM.

LLM bukan sumber kebenaran di sini: ia hanya menyusun kalimat dari evidence
yang sudah dikumpulkan pipeline, tidak pernah diminta mencari sumber, dan
selalu dipanggil dengan temperature rendah.

Provider disusun sebagai rantai fallback. Yang gagal ditandai tidak sehat untuk
sementara, supaya kredensial kedaluwarsa tidak membebani setiap permintaan
berikutnya dengan timeout yang sama.
"""

import logging
import os
from typing import List, Optional
from .. import config

logger = logging.getLogger(__name__)


# Model default untuk penalaran berbasis evidence: cepat, murah, dan mutakhir.
# Beban kerjanya bukan penalaran mendalam (bukti sudah diambil pipeline), jadi
# tier "mini" sudah memadai dan jauh lebih hemat untuk volume tinggi.
_DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

PROVIDER_OPENAI = "openai"
PROVIDER_COMPATIBLE = "openai_compatible"

# Berapa lama provider yang gagal dilewati sebelum dicoba lagi.
_UNHEALTHY_TTL_SECONDS = 300

# Fallback in-memory bila cache Django tidak tersedia (mis. saat skrip lepas).
_local_unhealthy: dict = {}


def _llm_enabled() -> bool:
    flag = config.get("INTELLIGENCE_LLM_ENABLED", "1")
    return str(flag).strip().lower() not in ("0", "false", "no", "off")


def openai_model() -> str:
    """Nama model OpenAI. Memakai `LLM_MODEL` yang sudah jadi konvensi repo ini."""
    return str(config.get("LLM_MODEL", _DEFAULT_OPENAI_MODEL) or _DEFAULT_OPENAI_MODEL)


def _uses_max_completion_tokens(model: str) -> bool:
    """
    Keluarga GPT-5 ke atas menolak `max_tokens` dan mensyaratkan
    `max_completion_tokens`. Deteksi ini menjaga kode tetap berfungsi baik pada
    model lama maupun baru tanpa perlu diubah lagi saat model diganti.
    """
    name = (model or "").lower().lstrip("gpt-")
    if not name:
        return False
    try:
        major = int(name.split(".")[0].split("-")[0])
    except (ValueError, IndexError):
        return (model or "").lower().startswith(("o1", "o3", "o4"))
    return major >= 5


def completion_token_kwargs(model: str, max_tokens: int) -> dict:
    """Nama parameter batas token yang sesuai untuk `model`."""
    key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
    return {key: max_tokens}


def _cache():
    try:
        from django.core.cache import cache
        return cache
    except Exception:  # pragma: no cover
        return None


def _unhealthy_key(provider: str) -> str:
    return f"llm_unhealthy:v1:{provider}"


def mark_unhealthy(provider: str, reason: str = "") -> None:
    """Tandai provider gagal agar dilewati sementara."""
    logger.warning("[LLM] Provider '%s' ditandai tidak sehat selama %ds: %s",
                   provider, _UNHEALTHY_TTL_SECONDS, reason[:200])
    cache = _cache()
    if cache is not None:
        try:
            cache.set(_unhealthy_key(provider), True, timeout=_UNHEALTHY_TTL_SECONDS)
            return
        except Exception:  # pragma: no cover
            pass
    import time
    _local_unhealthy[provider] = time.time() + _UNHEALTHY_TTL_SECONDS


def is_unhealthy(provider: str) -> bool:
    cache = _cache()
    if cache is not None:
        try:
            return bool(cache.get(_unhealthy_key(provider)))
        except Exception:  # pragma: no cover
            pass
    import time
    expiry = _local_unhealthy.get(provider)
    return bool(expiry and expiry > time.time())


def reset_health(provider: Optional[str] = None) -> None:
    """Bersihkan penandaan tidak sehat (dipakai test / setelah ganti kredensial)."""
    providers = [provider] if provider else [PROVIDER_OPENAI, PROVIDER_COMPATIBLE]
    cache = _cache()
    for name in providers:
        _local_unhealthy.pop(name, None)
        if cache is not None:
            try:
                cache.delete(_unhealthy_key(name))
            except Exception:  # pragma: no cover
                pass


def _compatible_key() -> str:
    return (config.get("LLM_API_KEY") or config.get("GROQ_API_KEY")
            or config.get("DEEPSEEK_API_KEY") or "")


def configured_providers() -> List[str]:
    """
    Provider yang punya kredensial, terurut berdasarkan preferensi.

    Dipilih lewat `LLM_PROVIDER` (variabel yang sudah dipakai pipeline
    training), mis. "openai". Nilainya eksklusif.
    """
    if not _llm_enabled():
        return []

    available = []
    if config.get("OPENAI_API_KEY"):
        available.append(PROVIDER_OPENAI)
    if _compatible_key():
        available.append(PROVIDER_COMPATIBLE)

    preference = str(config.get("LLM_PROVIDER", "") or "").strip().lower()
    if preference:
        # Preferensi bersifat EKSKLUSIF: menyebut "openai" berarti hanya OpenAI
        # yang dipakai. Ini yang diharapkan deployment yang memang sudah tidak
        # memakai penyedia lain, tidak ada percobaan sia-sia ke kredensial mati.
        wanted = [name.strip() for name in preference.split(",") if name.strip()]
        ordered = [name for name in wanted if name in available]
        missing = [name for name in wanted if name not in available]
        if missing:
            logger.warning(
                "[LLM] LLM_PROVIDER menyebut %s tetapi kredensialnya "
                "tidak ada; provider tersebut dilewati.", missing,
            )
        if not ordered:
            logger.error(
                "[LLM] Tidak ada provider yang cocok dengan LLM_PROVIDER=%r. "
                "Engine akan memakai jalur ekstraktif tanpa LLM.", preference,
            )
        return ordered

    return available


def available_provider() -> Optional[str]:
    """Provider pertama yang siap dipakai sekarang, atau None."""
    for provider in configured_providers():
        if not is_unhealthy(provider):
            return provider
    return None


def generate(prompt: str, *, temperature: float = 0.15,
             max_tokens: int = 900, system: str = "") -> Optional[str]:
    """
    Jalankan satu panggilan LLM dengan fallback antar provider.

    Mengembalikan None bila tidak ada provider yang berhasil, caller wajib
    punya jalur cadangan (lihat `reasoning.generator.build_extractive_answer`).
    """
    if not prompt:
        return None

    providers = configured_providers()
    if not providers:
        return None

    last_error = None
    tried = []

    for provider in providers:
        if is_unhealthy(provider):
            continue
        tried.append(provider)
        try:
            result = _generate_openai_compatible(
                prompt, temperature, max_tokens, system, provider
            )
            if result:
                if len(tried) > 1:
                    logger.info("[LLM] Berhasil lewat provider fallback '%s'", provider)
                return result
            logger.info("[LLM] Provider '%s' mengembalikan respons kosong", provider)
        except Exception as exc:
            last_error = exc
            mark_unhealthy(provider, f"{type(exc).__name__}: {exc}")

    if last_error is not None:
        logger.warning("[LLM] Semua provider gagal (%s); memakai jalur non-LLM. "
                       "Kesalahan terakhir: %s", tried, str(last_error)[:200])
    return None


def _generate_openai_compatible(prompt: str, temperature: float, max_tokens: int,
                                system: str, provider: str) -> Optional[str]:
    from openai import OpenAI

    if provider == PROVIDER_OPENAI:
        client = OpenAI(api_key=config.get("OPENAI_API_KEY"))
        model = openai_model()
    else:
        base_url = config.get("LLM_BASE_URL") or "https://api.groq.com/openai/v1"
        client = OpenAI(api_key=_compatible_key(), base_url=base_url)
        model = config.get("LLM_MODEL") or "llama-3.1-8b-instant"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **completion_token_kwargs(model, max_tokens),
    )
    content = response.choices[0].message.content if response.choices else ""
    return (content or "").strip() or None
