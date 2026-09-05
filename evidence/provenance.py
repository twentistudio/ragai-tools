"""
Claim provenance (§14) & summary provenance (§20).

Tujuan: sistem tahu **bagian mana dari response didukung sumber mana**, bukan
sekadar menempelkan daftar referensi di bawah jawaban.

Pendekatan sengaja deterministik (leksikal), bukan LLM, supaya:
  - tidak menambah biaya/latency,
  - tidak bisa "mengarang" atribusi,
  - mudah diuji.
"""

import re
from typing import Any, Dict, Iterable, List, Optional

from ..contracts import EvidenceItem, Provenance, SupportedClaim

# Kata yang terlalu umum untuk dipakai sebagai bukti kecocokan.
_STOPWORDS = {
    # Indonesia
    "yang", "dan", "atau", "dengan", "untuk", "pada", "dari", "ini", "itu",
    "adalah", "akan", "dapat", "bisa", "tidak", "juga", "oleh", "dalam",
    "sebagai", "karena", "jika", "agar", "lebih", "sudah", "telah", "masih",
    "anda", "saya", "kami", "mereka", "namun", "tetapi", "serta", "secara",
    "seperti", "hingga", "sehingga", "antara", "terhadap", "bahwa", "para",
    # English
    "the", "and", "or", "with", "for", "from", "this", "that", "these",
    "those", "are", "was", "were", "have", "has", "had", "can", "could",
    "will", "would", "should", "may", "might", "not", "but", "also", "your",
    "you", "they", "them", "their", "there", "which", "what", "when", "than",
    "into", "such", "been", "being", "more", "most", "some", "other",
}

# Pemisahan kalimat harus MENAHAN penanda sitasi pada kalimat pemiliknya:
# pada "... perempuan. [E1] Gejalanya ...", penanda [E1] menerangkan kalimat
# pertama. Memisah tepat sebelum "[E1]" akan memindahkannya ke kalimat kedua —
# kalimat yang benar-benar bersumber jadi tampak tidak bersumber.
_SENTENCE_SCAN_RE = re.compile(r"(.+?[.!?]+(?:\s*\[E\d{1,2}\])*)(?=\s+|$)")
_CITATION_RE = re.compile(r"\[E(\d{1,2})\]")
_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)

# Kalimat yang jelas bukan pernyataan faktual medis (disclaimer, ajakan, dsb).
_NON_FACTUAL_RE = re.compile(
    r"\b(konsultasikan|segera periksa|hubungi dokter|bukan diagnosis|"
    r"informasi ini|disclaimer|silakan|sebaiknya anda|semoga|terima kasih|"
    r"this is not|consult a|please )\b",
    re.IGNORECASE,
)

# Kalimat yang dihasilkan sistem sendiri (template, peringatan, catatan) —
# bukan pernyataan medis, jadi tidak perlu dicarikan dukungan bukti.
_SYSTEM_TEXT_RE = re.compile(
    r"(sistem ini|sistem tidak|kami tidak menebak|kami belum menemukan|"
    r"basis pengetahuan|di luar cakupan|peringatan:|langkah yang disarankan|"
    r"catatan:|tidak ditemukan bukti|gawat darurat|ringkasan ini disusun|"
    r"berikut informasi dari literatur|berikut bukti yang ditemukan|"
    r"melanjutkan pembahasan|untuk pilihan dan dosis obat|"
    # kalimat meta tentang proses penilaian, bukan pernyataan medis
    r"belum (?:dapat dianggap )?final|bukti yang tersedia terbatas|"
    # ungkapan ketidakpastian penulis — bukan pernyataan medis yang perlu
    # ditelusuri ke sumber, justru kebalikannya
    r"belum (?:bisa|dapat) (?:memastikan|disimpulkan|dipastikan)|"
    r"tidak (?:bisa|dapat) (?:memastikan|dipastikan|disimpulkan)|"
    r"sulit dipastikan|masih perlu (?:dipastikan|dikonfirmasi)|"
    r"hanya dari (?:bukti|informasi) ini|dari bukti ini saja|"
    r"perlu pemeriksaan lebih lanjut|"
    r"belum memadai|tidak memadai|belum cukup untuk menjawab|"
    r"jika anda ingin|saya bisa (?:bantu|membantu)|kalau anda mau|"
    r"klaim (?:ini|tersebut) (?:tidak )?didukung|"
    r"tidak didukung oleh (?:bukti|evidence)|"
    r"evidence yang (?:ada|tersedia)|"
    r"penilaian ini|kesimpulan di atas)",
    re.IGNORECASE,
)

MIN_OVERLAP_RATIO = 0.22
MIN_OVERLAP_TERMS = 2


def _tokens(text: str) -> set:
    return {
        w.lower() for w in _WORD_RE.findall(text or "")
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }


def split_sentences(text: str) -> List[str]:
    """Pecah jawaban menjadi kalimat, buang bullet marker dan spasi berlebih."""
    if not text:
        return []

    raw_sentences: List[str] = []
    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue
        position = 0
        for match in _SENTENCE_SCAN_RE.finditer(line):
            raw_sentences.append(match.group(1))
            position = match.end()
        tail = line[position:].strip()
        if tail:
            raw_sentences.append(tail)

    parts: List[str] = []
    for raw in raw_sentences:
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw or "").strip()
        if len(cleaned) >= 15:
            parts.append(cleaned)
    return parts


def is_factual_sentence(sentence: str) -> bool:
    """Kalimat yang layak dicarikan dukungan bukti."""
    if not sentence or len(sentence) < 20:
        return False
    if _NON_FACTUAL_RE.search(sentence):
        return False
    if _SYSTEM_TEXT_RE.search(sentence):
        return False
    if sentence.strip().endswith("?"):
        return False
    return len(_tokens(sentence)) >= 3


def _reference(item: EvidenceItem, **extra) -> Dict[str, Any]:
    ref = {
        "chunk_id": item.chunk_id,
        "source_id": item.source_id or item.chunk_id,
        "title": item.title,
        "doi": item.doi or None,
        "url": item.url or None,
    }
    ref.update(extra)
    return ref


def attribute_claims(answer: str, evidence: Iterable[EvidenceItem],
                     max_claims: int = 8) -> List[SupportedClaim]:
    """
    Petakan tiap pernyataan faktual di `answer` ke evidence pendukungnya.

    Dua sinyal dipakai, berurutan:
      1. **Penanda sitasi `[En]`** yang ditulis response engine. Ini sinyal
         utama dan tahan beda bahasa (jawaban Bahasa Indonesia, evidence
         berbahasa Inggris).
      2. **Tumpang tindih leksikal** dengan judul/cuplikan evidence, sebagai
         cadangan bila penanda tidak ada.

    Kalimat tanpa dukungan tetap dikembalikan dengan
    `verdict = "unsupported"` dan `supporting_evidence = []`, sehingga
    caller (safety layer) bisa mendeteksi klaim yang tidak berdasar.
    """
    evidence_list = [e for e in (evidence or [])]
    results: List[SupportedClaim] = []

    ev_tokens = [(e, _tokens(f"{e.title} {e.snippet}")) for e in evidence_list]

    for raw_sentence in split_sentences(answer):
        # Penanda sitasi diambil dulu, lalu dibersihkan dari teks klaim.
        cited_indexes = [int(m.group(1)) for m in _CITATION_RE.finditer(raw_sentence)]
        sentence = _CITATION_RE.sub("", raw_sentence).strip()
        sentence = re.sub(r"\s+([.,;])", r"\1", sentence)
        sentence = re.sub(r"\s{2,}", " ", sentence).strip()

        if not is_factual_sentence(sentence):
            continue

        supporters: List[Dict[str, Any]] = []

        # 1) sitasi eksplisit
        for index in cited_indexes:
            if 1 <= index <= len(evidence_list):
                supporters.append(_reference(
                    evidence_list[index - 1], via="citation_marker", overlap=None,
                    matched_terms=[],
                ))

        # 2) tumpang tindih leksikal
        if not supporters:
            s_tokens = _tokens(sentence)
            if not s_tokens:
                continue
            lexical: List[Dict[str, Any]] = []
            for item, tokens in ev_tokens:
                if not tokens:
                    continue
                shared = s_tokens & tokens
                ratio = len(shared) / float(len(s_tokens))
                if len(shared) >= MIN_OVERLAP_TERMS and ratio >= MIN_OVERLAP_RATIO:
                    lexical.append(_reference(
                        item, via="lexical_overlap", overlap=round(ratio, 3),
                        matched_terms=sorted(shared)[:8],
                    ))
            lexical.sort(key=lambda s: s["overlap"], reverse=True)
            supporters = lexical

        confidence = None
        if supporters:
            first = supporters[0]
            confidence = 0.85 if first.get("via") == "citation_marker" else round(
                min(1.0, (first.get("overlap") or 0.0) * 2.0), 3
            )

        results.append(SupportedClaim(
            claim=sentence,
            supporting_evidence=supporters[:3],
            verdict="supported" if supporters else "unsupported",
            confidence=confidence,
        ))

        if len(results) >= max_claims:
            break

    return results


def unsupported_claims(claims: Iterable[SupportedClaim]) -> List[SupportedClaim]:
    return [c for c in (claims or []) if c.verdict == "unsupported"]


def provenance_entry(value: Any, source: Provenance,
                     detail: Optional[str] = None) -> Dict[str, Any]:
    """Bungkus sebuah nilai summary dengan asal-usulnya (§20)."""
    return {
        "value": value,
        "provenance": source.value if isinstance(source, Provenance) else str(source),
        "detail": detail,
    }
