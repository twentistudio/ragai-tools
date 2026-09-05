"""
Penanda sitasi internal.

Penanda `[E1]` menelusuri kalimat ke bukti nomor berapa, tetapi di layar hanya
tampak sebagai angka dalam kurung. Satu definisi di sini dipakai seluruh jalur
yang menghasilkan teks siap tampil.
"""

import re

CITATION_MARKER_RE = re.compile(r"\s*\[E\d{1,2}\]")


def strip_citation_markers(text: str) -> str:
    """Buang penanda sitasi dan rapikan spasi yang ditinggalkannya."""
    cleaned = CITATION_MARKER_RE.sub("", text or "")
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
