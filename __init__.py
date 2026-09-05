"""
ragai — mesin tanya jawab kesehatan berbasis literatur jurnal.

Menerima satu pertanyaan kesehatan dan mengembalikan jawaban beserta jurnal
yang menopangnya. Setiap DOI diperiksa ke registry sebelum disajikan, dan
ketika literaturnya tidak memuat topik yang ditanyakan, engine menyatakannya
alih-alih menyusun jawaban yang terdengar masuk akal.

Engine tidak mengenal aplikasi yang memakainya. Kebutuhannya terhadap
penyimpanan dan layanan luar disebutkan lewat nama peran di `runtime`, dan
aplikasi induk memenuhinya saat start:

    from ragai import runtime

    runtime.configure(
        models={"JournalArticle": ..., "ConversationSession": ...},
        services={"translate": ..., "embed_article": ...},
    )

Sesudah itu satu panggilan sudah cukup:

    import ragai

    hasil = ragai.process({"query": "apakah demam berdarah ditularkan nyamuk"})

Dipakai dua arah: produk Healthify memanggilnya sebagai pustaka dalam proses
yang sama, sementara konsumen luar memanggilnya lewat HTTP di
`/api/v1/intelligence/`.
"""

from . import runtime
from .contracts import (
    EvidenceStatus,
    Intent,
    IntelligenceRequest,
    IntelligenceResponse,
    Mode,
    SafetyDecision,
)

__all__ = [
    "EvidenceStatus",
    "Intent",
    "IntelligenceRequest",
    "IntelligenceResponse",
    "Mode",
    "SafetyDecision",
    "process",
    "runtime",
    "version",
]


def process(payload, consumer="host"):
    """
    Jawab satu pertanyaan kesehatan.

    `engine` diimpor di dalam fungsi karena ia menarik lapisan retrieval dan
    penyimpanan; mengimpornya saat modul dimuat memaksa Django siap sebelum
    `ragai` boleh disentuh sama sekali.
    """
    from .engine import process as _process

    return _process(payload, consumer=consumer)


def version() -> str:
    """Versi engine, sama dengan yang dilaporkan endpoint capabilities."""
    from .engine import ENGINE_VERSION

    return ENGINE_VERSION
