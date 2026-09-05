"""
Embedding teks yang tidak terikat satu penyedia.

Latar
-----
Pipeline training Healthify memakai Gemini (`embed_texts_gemini`, 768 dimensi)
dan tabel pgvector dibuat dengan dimensi tersebut. Bila deployment tidak lagi
memakai Gemini, jalur embedding mati total: admin tidak bisa meng-embed jurnal
dan retrieval semantik tidak jalan.

Modul ini menyediakan embedding lewat penyedia yang benar-benar tersedia,
**dengan dimensi yang cocok dengan kolom vektor yang sudah ada**, sehingga
tidak perlu migrasi tabel maupun re-embed seluruh korpus.

Urutan penyedia:
    1. OpenAI  `text-embedding-3-small` — mendukung parameter `dimensions`,
       jadi keluarannya bisa dipotong tepat ke dimensi tabel (mis. 768).
    2. Gemini  `embed_texts_gemini` dari pipeline training (bila tersedia).
    3. sentence-transformers (bila modul training terpasang).

Bila tidak satu pun tersedia, fungsi mengembalikan None — caller harus
menanganinya, bukan menyimpan vektor nol.
"""

import logging
import os
from typing import List, Optional
from .. import runtime

logger = logging.getLogger(__name__)

DEFAULT_DIMENSIONS = 768
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

_cached_dimension: Optional[int] = None


def _setting(name: str, default=None):
    try:
        from django.conf import settings
        value = getattr(settings, name, None)
        if value not in (None, ""):
            return value
    except Exception:  # pragma: no cover
        pass
    return os.getenv(name, default)


def target_dimensions() -> int:
    """
    Dimensi vektor yang harus dihasilkan.

    Urutan penentuan:
      1. setting/env `EMBEDDING_DIMENSIONS`
      2. dimensi kolom vektor yang sudah terisi di database
      3. DEFAULT_DIMENSIONS (768, sesuai pipeline training)
    """
    global _cached_dimension

    configured = _setting("EMBEDDING_DIMENSIONS")
    if configured:
        try:
            return int(configured)
        except (TypeError, ValueError):
            pass

    if _cached_dimension is not None:
        return _cached_dimension

    detected = _detect_dimension_from_db()
    _cached_dimension = detected or DEFAULT_DIMENSIONS
    return _cached_dimension


def _detect_dimension_from_db() -> Optional[int]:
    """Baca dimensi dari satu baris vektor yang sudah ada."""
    try:
        import sys

        scripts_dir = runtime.service("training_scripts_dir")
        TRAINING_SCRIPTS_DIR = scripts_dir() if scripts_dir else None
        if TRAINING_SCRIPTS_DIR and str(TRAINING_SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(TRAINING_SCRIPTS_DIR))
        from ingest_chunks_to_pg import DB_TABLE, connect_db  # type: ignore

        conn = connect_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT vector_dims(embedding) FROM {DB_TABLE} "
                    f"WHERE embedding IS NOT NULL LIMIT 1;"
                )
                row = cur.fetchone()
                if row and row[0]:
                    logger.info("[EMBED] Dimensi vektor terdeteksi dari database: %s", row[0])
                    return int(row[0])
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("[EMBED] Deteksi dimensi dari database gagal: %s", exc)
    return None


def reset_dimension_cache() -> None:
    global _cached_dimension
    _cached_dimension = None


def embeddings_enabled() -> bool:
    """
    Gerbang untuk mematikan pemanggilan API embedding.

    Dipakai oleh test suite agar tidak menembak jaringan, dan berguna saat
    deployment ingin menjalankan retrieval leksikal saja.
    """
    flag = _setting("EMBEDDINGS_ENABLED", "1")
    return str(flag).strip().lower() not in ("0", "false", "no", "off")


def available_provider() -> Optional[str]:
    if not embeddings_enabled():
        return None
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API"):
        return "gemini"
    try:
        available = runtime.service("training_modules_available")
        if available and available():
            return "sentence_transformers"
    except Exception:  # pragma: no cover
        pass
    return None


def embed_texts(texts: List[str], dimensions: Optional[int] = None) -> Optional[List[List[float]]]:
    """
    Hasilkan embedding untuk sekumpulan teks.

    Returns list vektor, atau None bila tidak ada penyedia yang berhasil.
    """
    texts = [t for t in (texts or []) if t and str(t).strip()]
    if not texts:
        return []

    if not embeddings_enabled():
        return None

    dims = int(dimensions or target_dimensions())

    if os.getenv("OPENAI_API_KEY"):
        try:
            return _embed_openai(texts, dims)
        except Exception as exc:
            logger.warning("[EMBED] OpenAI embedding gagal: %s", exc)

    try:
        return _embed_training(texts, dims)
    except Exception as exc:
        logger.info("[EMBED] Embedding pipeline training tidak tersedia: %s", exc)

    return None


def embed_text(text: str, dimensions: Optional[int] = None) -> Optional[List[float]]:
    """Embedding untuk satu teks."""
    vectors = embed_texts([text], dimensions=dimensions)
    return vectors[0] if vectors else None


def _embed_openai(texts: List[str], dims: int) -> List[List[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = _setting("EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)

    # Batasi panjang input; model punya batas token per item.
    trimmed = [str(t)[:8000] for t in texts]

    response = client.embeddings.create(
        model=model,
        input=trimmed,
        dimensions=dims,  # memotong vektor agar cocok dengan kolom yang ada
    )
    vectors = [item.embedding for item in response.data]

    if vectors and len(vectors[0]) != dims:
        raise ValueError(
            f"OpenAI mengembalikan {len(vectors[0])} dimensi, diharapkan {dims}"
        )
    logger.info("[EMBED] %d embedding dibuat lewat OpenAI (%s, %d dim)",
                len(vectors), model, dims)
    return vectors


def _embed_training(texts: List[str], dims: int) -> List[List[float]]:
    import sys

    scripts_dir = runtime.service("training_scripts_dir")
    TRAINING_SCRIPTS_DIR = scripts_dir() if scripts_dir else None
    if TRAINING_SCRIPTS_DIR and str(TRAINING_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_SCRIPTS_DIR))
    from chunk_and_embed import embed_texts_gemini  # type: ignore

    vectors = embed_texts_gemini(list(texts))
    if not vectors:
        raise ValueError("embed_texts_gemini mengembalikan hasil kosong")
    if len(vectors[0]) != dims:
        raise ValueError(
            f"Pipeline training mengembalikan {len(vectors[0])} dimensi, diharapkan {dims}"
        )
    return vectors


# ---------------------------------------------------------------------------
# Penyediaan tabel vektor
# ---------------------------------------------------------------------------

VECTOR_TABLE = "embeddings"


def ensure_vector_store(dimensions: Optional[int] = None) -> bool:
    """
    Pastikan ekstensi pgvector dan tabel `embeddings` tersedia.

    Deployment yang tidak pernah menjalankan pipeline ingest training belum
    punya keduanya, sehingga penyimpanan embedding gagal diam-diam. Fungsi ini
    membuatnya sesuai skema `training/scripts/ingest_chunks_to_pg.ensure_table`
    agar kedua jalur tetap kompatibel.

    Returns True bila vector store siap dipakai.
    """
    dims = int(dimensions or target_dimensions())
    try:
        from django.db import connection

        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (
                    id bigserial PRIMARY KEY,
                    doc_id text,
                    safe_id text,
                    source_file text,
                    chunk_index integer,
                    n_words integer,
                    text text,
                    doi text,
                    embedding vector({dims}),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {VECTOR_TABLE}_doc_id_idx "
                f"ON {VECTOR_TABLE} (doc_id);"
            )
        logger.info("[EMBED] Vector store siap (%s, %d dim)", VECTOR_TABLE, dims)
        return True
    except Exception as exc:
        logger.warning("[EMBED] Vector store tidak dapat disiapkan: %s", exc)
        return False


def store_vector(doc_id: str, safe_id: str, source_file: str, text: str,
                 doi: str, embedding: list, dimensions: Optional[int] = None) -> bool:
    """
    Simpan satu vektor ke tabel `embeddings`.

    Best-effort: kegagalan di sini TIDAK boleh menggagalkan operasi pemanggil,
    karena retrieval leksikal + embedding yang tersimpan di kolom Django sudah
    cukup untuk menjawab. Returns True bila tersimpan.
    """
    if not embedding:
        return False
    if not ensure_vector_store(dimensions or len(embedding)):
        return False

    try:
        from django.db import connection

        emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
        with connection.cursor() as cursor:
            cursor.execute(f"DELETE FROM {VECTOR_TABLE} WHERE doc_id = %s;", (doc_id,))
            cursor.execute(
                f"""INSERT INTO {VECTOR_TABLE}
                    (doc_id, safe_id, source_file, chunk_index, n_words, text, doi, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)""",
                (doc_id, safe_id, source_file, 0, len(text.split()), text, doi or "", emb_str),
            )
        return True
    except Exception as exc:
        logger.warning("[EMBED] Gagal menyimpan vektor %s: %s", doc_id, exc)
        return False
