"""
Kontrak internal Health Intelligence Engine.

Modul ini hanya berisi struktur data. Tidak ada I/O, tidak ada dependency ke
Django ORM, sehingga bisa dipakai/di-test secara terisolasi.

Catatan kompatibilitas:
    Kontrak di sini adalah kontrak *internal*. Public API Healthify yang sudah
    ada (POST /api/verify/ dsb) tidak memakai struktur ini secara langsung, penerjemahan dilakukan oleh adapter di `api.intelligence.adapters`.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Intent(str, Enum):
    """Kebutuhan informasi user."""

    CLAIM_VERIFICATION = "CLAIM_VERIFICATION"
    HEALTH_INFORMATION = "HEALTH_INFORMATION"
    SYMPTOM_CONTEXT = "SYMPTOM_CONTEXT"
    FOLLOW_UP = "FOLLOW_UP"
    MEDICATION_INFORMATION = "MEDICATION_INFORMATION"
    GENERAL_HEALTH = "GENERAL_HEALTH"
    # Sapaan, ucapan terima kasih, dan penutup percakapan. Dipisahkan dari
    # UNSUPPORTED karena maknanya berbeda: pesan seperti ini bukan pertanyaan
    # di luar cakupan, melainkan basa-basi yang wajar di ruang obrolan dan
    # pantas dijawab ramah, bukan dijawab dengan lima jurnal.
    SMALL_TALK = "SMALL_TALK"
    UNSUPPORTED = "UNSUPPORTED"


class Mode(str, Enum):
    """Mode pemakaian engine."""

    CLAIM = "claim"
    CONSULTATION = "consultation"
    INFORMATION = "information"
    MEDICATION = "medication"

    @classmethod
    def coerce(cls, value: Any, default: "Mode" = None) -> "Mode":
        default = default or cls.CONSULTATION
        if isinstance(value, cls):
            return value
        if not value:
            return default
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return default


class EvidenceStatus(str, Enum):
    """Hasil tahap evidence retrieval."""

    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SafetyDecision(str, Enum):
    """Keputusan safety validator."""

    PASS = "PASS"
    MODIFY = "MODIFY"
    BLOCK = "BLOCK"


class Provenance(str, Enum):
    """Asal-usul sebuah informasi dalam summary/response."""

    USER_REPORTED = "USER_REPORTED"
    AI_INFERRED = "AI_INFERRED"
    EVIDENCE_SUPPORTED = "EVIDENCE_SUPPORTED"
    SYSTEM_GENERATED = "SYSTEM_GENERATED"


class EvidenceOrigin(str, Enum):
    """
    Dari mana sebuah evidence item berasal. Ini adalah pertahanan utama
    terhadap DOI halusinasi: hanya evidence dengan origin
    KNOWLEDGE_BASE / VECTOR_INDEX / VERIFIED_REGISTRY yang boleh
    dipublikasikan sebagai sumber.
    """

    KNOWLEDGE_BASE = "KNOWLEDGE_BASE"      # JournalArticle / Source di DB Healthify
    VECTOR_INDEX = "VECTOR_INDEX"          # tabel embeddings (pgvector)
    VERIFIED_REGISTRY = "VERIFIED_REGISTRY"  # DOI diverifikasi ke doi.org/Crossref
    MODEL_SUGGESTED = "MODEL_SUGGESTED"    # dikarang LLM -> TIDAK boleh dipublikasikan
    USER_SUPPLIED = "USER_SUPPLIED"        # dari dispute/report user


# Origin yang boleh keluar sebagai sumber ke user.
PUBLISHABLE_ORIGINS = frozenset({
    EvidenceOrigin.KNOWLEDGE_BASE,
    EvidenceOrigin.VECTOR_INDEX,
    EvidenceOrigin.VERIFIED_REGISTRY,
    EvidenceOrigin.USER_SUPPLIED,
})


def _clean(value: Any) -> str:
    return (str(value).strip() if value not in (None, "") else "")


@dataclass
class HealthContext:
    """Structured health context. Field yang tidak dilaporkan user = None/[]."""

    chief_complaint: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    duration: Optional[str] = None
    severity: Optional[str] = None
    onset: Optional[str] = None
    progression: Optional[str] = None
    associated_symptoms: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    relevant_history: List[str] = field(default_factory=list)
    # peta field -> Provenance, supaya summary bisa menjelaskan asal data
    provenance: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_empty(self) -> bool:
        return not any([
            self.chief_complaint, self.symptoms, self.duration, self.severity,
            self.onset, self.progression, self.associated_symptoms,
            self.medications, self.allergies, self.relevant_history,
        ])

    def merge(self, other: "HealthContext") -> "HealthContext":
        """Gabungkan context lama dengan context baru (- akumulasi gejala).

        Nilai baru menang untuk field skalar; list digabung tanpa duplikat.
        """
        def pick(new_v, old_v):
            return new_v if new_v not in (None, "") else old_v

        def union(new_l, old_l):
            merged = list(old_l or [])
            for item in (new_l or []):
                if item not in merged:
                    merged.append(item)
            return merged

        merged_prov = dict(self.provenance or {})
        merged_prov.update(other.provenance or {})

        return HealthContext(
            chief_complaint=pick(other.chief_complaint, self.chief_complaint),
            symptoms=union(other.symptoms, self.symptoms),
            duration=pick(other.duration, self.duration),
            severity=pick(other.severity, self.severity),
            onset=pick(other.onset, self.onset),
            progression=pick(other.progression, self.progression),
            associated_symptoms=union(other.associated_symptoms, self.associated_symptoms),
            medications=union(other.medications, self.medications),
            allergies=union(other.allergies, self.allergies),
            relevant_history=union(other.relevant_history, self.relevant_history),
            provenance=merged_prov,
        )


@dataclass
class EvidenceItem:
    """Satu potongan evidence yang sudah tervalidasi."""

    chunk_id: str = ""
    source_id: str = ""
    title: str = ""
    snippet: str = ""
    doi: str = ""
    url: str = ""
    authors: str = ""
    publisher: str = ""
    published_year: Optional[int] = None
    source_type: str = "journal"
    origin: EvidenceOrigin = EvidenceOrigin.KNOWLEDGE_BASE

    # skor
    semantic_relevance: float = 0.0
    source_quality: float = 0.5
    publication_recency: float = 0.5
    evidence_type_score: float = 0.5
    context_match: float = 0.0
    # Seberapa banyak ASPEK yang ditanyakan (gejala? pengobatan? penularan?)
    # benar-benar dibahas dokumen ini. 1.0 bila pertanyaan tidak menyebut aspek
    # tertentu. Mencocokkan nama penyakit saja tidak menjadikan sebuah paper
    # sebagai jawaban.
    aspect_match: float = 1.0
    # True bila JUDUL dokumen jelas membahas topik lain. Dokumen semacam ini
    # bisa saja memuat kata kunci yang sama di dalam abstraknya, tetapi
    # menyajikannya sebagai bukti hanya membuat pembaca membuka paper yang tidak
    # ada kaitannya dengan pertanyaannya.
    off_topic: bool = False
    relevance: float = 0.0  # skor gabungan hasil re-ranking

    # status validasi link (/ anti-404)
    doi_verified: bool = False
    link_status: str = "unchecked"  # unchecked | verified | unresolvable | skipped
    # True bila judul yang dikirim ternyata milik karya lain dan diganti dengan
    # judul resmi dari registry.
    title_corrected: bool = False

    def is_publishable(self) -> bool:
        """
        Boleh ditampilkan ke user sebagai sumber?

        Aturan:
        - Origin MODEL_SUGGESTED (dikarang LLM) hanya lolos bila DOI-nya
          sudah diverifikasi ke registry resmi.
        - Link yang dipastikan mati / format DOI ngawur langsung ditolak.
        - Harus punya minimal judul atau cuplikan teks.
        """
        if self.origin not in PUBLISHABLE_ORIGINS and not self.doi_verified:
            return False
        if self.link_status in ("unresolvable", "malformed"):
            return False
        return bool(self.title or self.snippet)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["origin"] = self.origin.value if isinstance(self.origin, EvidenceOrigin) else str(self.origin)
        return data

    def to_public_dict(self) -> Dict[str, Any]:
        """Bentuk ringkas untuk response API."""
        return {
            "source_id": self.source_id or self.chunk_id,
            "chunk_id": self.chunk_id,
            "title": self.title,
            "doi": self.doi or None,
            "url": self.url or None,
            "publisher": self.publisher or None,
            "published_year": self.published_year,
            "source_type": self.source_type,
            "relevance": round(float(self.relevance), 4),
            "snippet": self.snippet[:400],
            "origin": self.origin.value if isinstance(self.origin, EvidenceOrigin) else str(self.origin),
            "doi_verified": bool(self.doi_verified),
            "link_status": self.link_status,
        }


@dataclass
class SupportedClaim:
    """Claim provenance: pernyataan -> evidence pendukung."""

    claim: str = ""
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    verdict: Optional[str] = None       # supported | unsupported | inconclusive
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyFlag:
    """Satu temuan lapisan keamanan: kode, tingkat, dan pesannya."""

    code: str
    severity: str = "info"  # info | warning | critical
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntelligenceRequest:
    """Unified internal request."""

    query: str = ""
    mode: Mode = Mode.CONSULTATION
    conversation_id: Optional[str] = None
    previous_messages: List[Dict[str, str]] = field(default_factory=list)
    health_context: Dict[str, Any] = field(default_factory=dict)
    include_evidence: bool = True
    include_sources: bool = True
    language: str = "id"
    # Delapan, bukan lima. Jawaban yang bertumpu pada dua-tiga paper mudah
    # meleset ketika kebetulan yang terambil membahas sisi lain dari topik;
    # basis rujukan yang lebih lebar membuat kesimpulan lebih stabil.
    max_evidence: int = 8
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "IntelligenceRequest":
        payload = payload or {}
        context = payload.get("context") or {}
        options = payload.get("options") or {}

        raw_messages = context.get("previous_messages") or []
        messages: List[Dict[str, str]] = []
        for msg in raw_messages:
            if isinstance(msg, dict):
                role = _clean(msg.get("role") or msg.get("sender") or "user").lower()
                content = _clean(msg.get("content") or msg.get("text") or msg.get("message"))
                if content:
                    messages.append({
                        "role": "assistant" if role in ("assistant", "system", "bot", "ai") else "user",
                        "content": content,
                    })
            elif isinstance(msg, str) and msg.strip():
                messages.append({"role": "user", "content": msg.strip()})

        try:
            max_evidence = int(options.get("max_evidence", 8))
        except (TypeError, ValueError):
            max_evidence = 8
        max_evidence = max(1, min(max_evidence, 20))

        return cls(
            query=_clean(payload.get("query") or payload.get("text") or payload.get("message")),
            mode=Mode.coerce(payload.get("mode")),
            # Pengenal ruang obrolan datang dari produk lain, dan tiap produk
            # menamainya berbeda. Menerima beberapa nama membuat konsumen tidak
            # perlu menyesuaikan skema datanya hanya untuk memanggil API ini;
            # nilai apa pun yang mereka sudah punya bisa langsung dipakai.
            conversation_id=_clean(
                context.get("conversation_id")
                or context.get("session_id")
                or context.get("room_id")
                or context.get("thread_id")
                or context.get("chat_id")
                or payload.get("conversation_id")
                or payload.get("session_id")
            ) or None,
            previous_messages=messages,
            health_context=context.get("health_context") or {},
            include_evidence=bool(options.get("include_evidence", True)),
            include_sources=bool(options.get("include_sources", True)),
            language=_clean(options.get("language") or "id").lower() or "id",
            max_evidence=max_evidence,
            metadata=payload.get("metadata") or {},
        )


@dataclass
class IntelligenceResponse:
    """Standardized internal response."""

    answer: str = ""
    intent: Intent = Intent.GENERAL_HEALTH
    mode: Mode = Mode.CONSULTATION
    health_context: HealthContext = field(default_factory=HealthContext)
    evidence: List[EvidenceItem] = field(default_factory=list)
    claims: List[SupportedClaim] = field(default_factory=list)
    uncertainty: Optional[str] = None
    evidence_status: EvidenceStatus = EvidenceStatus.INSUFFICIENT_EVIDENCE
    safety_decision: SafetyDecision = SafetyDecision.PASS
    safety_flags: List[SafetyFlag] = field(default_factory=list)
    preliminary_assessment: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_evidence: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "answer": self.answer,
            "intent": self.intent.value if isinstance(self.intent, Intent) else str(self.intent),
            "mode": self.mode.value if isinstance(self.mode, Mode) else str(self.mode),
            "health_context": self.health_context.to_dict(),
            "evidence": [e.to_public_dict() for e in self.evidence] if include_evidence else [],
            "claims": [c.to_dict() for c in self.claims],
            "uncertainty": self.uncertainty,
            "evidence_status": self.evidence_status.value,
            "safety_decision": self.safety_decision.value,
            "safety_flags": [f.to_dict() for f in self.safety_flags],
            "preliminary_assessment": self.preliminary_assessment,
            "conversation_id": self.conversation_id,
            "metadata": self.metadata,
        }
        return data


# Disclaimer yang menyertai setiap asesmen awal
PRELIMINARY_ASSESSMENT_STATUS = "PRELIMINARY_ASSESSMENT"
PRELIMINARY_ASSESSMENT_DISCLAIMER = (
    "Informasi ini merupakan asesmen awal yang dihasilkan AI (AI-generated "
    "preliminary information) dan BUKAN diagnosis medis. Konsultasikan dengan "
    "tenaga kesehatan untuk penilaian dan penanganan yang tepat."
)
