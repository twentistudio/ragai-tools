"""
ragai, mesin tanya jawab kesehatan berbasis literatur jurnal.

Menerima satu pertanyaan, mengembalikan jawaban beserta jurnal yang
menopangnya. Ketika literaturnya tidak memuat topik yang ditanyakan, engine
menyatakannya alih-alih menyusun jawaban yang terdengar masuk akal.

Aplikasi induk mendaftarkan penyimpanan dan layanan yang dibutuhkan lewat
`runtime.configure`; sesudah itu `ragai.process(payload)` sudah cukup. Rincian
pemasangan ada di README.md.
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
