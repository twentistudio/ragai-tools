# ragai

Mesin tanya jawab kesehatan berbasis literatur jurnal. Menerima satu
pertanyaan, mengembalikan jawaban beserta jurnal yang menopangnya. Setiap DOI
diperiksa ke registry sebelum disajikan, dan ketika literaturnya tidak memuat
topik yang ditanyakan, engine menyatakannya alih-alih menyusun jawaban yang
terdengar masuk akal.

## Batas dengan aplikasi induk

Engine tidak mengimpor apa pun dari aplikasi yang memakainya. Yang dibutuhkan
disebutkan lewat nama peran di `runtime`, dan aplikasi induk yang memenuhinya.
Batas ini dijaga uji `EngineBoundaryTests`, sehingga satu impor yang bocor
langsung terlihat.

Yang harus didaftarkan:

| Peran | Dipakai untuk |
|-|-|
| `JournalArticle` | basis pengetahuan jurnal |
| `Source`, `ClaimSource` | sumber yang pernah terkait dengan klaim |
| `ConversationSession`, `ConversationMessage` | percakapan multi-giliran |
| `ConsultationSummary` | ringkasan konsultasi |

Layanan berikut opsional. Tanpa `translate`, kosakata di luar leksikon tidak
terjangkau; tanpa `embed_article`, jurnal baru hanya ditemukan lewat kata kunci.

| Layanan | Bentuk |
|-|-|
| `translate` | `(teks, bahasa_tujuan) -> teks` |
| `embed_article` | `(artikel) -> None` |
| `training_scripts_dir` | `() -> Path` |
| `training_modules_available` | `() -> bool` |

## Dua cara pemakaian

**Sebagai pustaka**, di dalam proses yang sama. Inilah yang dipakai Healthify
untuk memverifikasi klaim:

```python
import ragai

hasil = ragai.process({"query": "apakah demam berdarah ditularkan nyamuk"})
print(hasil.answer, len(hasil.evidence))
```

**Lewat HTTP**, untuk aplikasi lain. Dilayani `api/intelligence_views.py` di
`/api/v1/intelligence/`, dengan kunci API per konsumen dan batas laju
tersendiri. Dokumentasinya di `https://ragai.twenti.studio/docs`.

## Memasang di aplikasi lain

Salin direktori ini, lalu daftarkan pemenuh perannya saat aplikasi start. Pada
Django, tempatnya di `AppConfig.ready()`:

```python
from ragai import runtime

runtime.configure(
    models={"JournalArticle": JournalArticle, ...},
    services={"translate": terjemahkan, "embed_article": buat_embedding},
)
```

Model harus menyediakan field yang dipakai engine (`title`, `abstract`, `doi`,
`url`, `embedding`, `is_embedded` untuk jurnal). Basis datanya masih milik
aplikasi induk: memindahkan tabel ke engine berarti migrasi rename, dan itu
ditunda sampai engine benar-benar dipisahkan ke repositori sendiri.

## Memisahkan ke repositori sendiri

Kode engine sudah bersih dari aplikasi induk, tetapi belum semua miliknya
berada di direktori ini. Yang berikut masih di sisi Healthify dan ikut pindah
saat pemisahan sungguhan:

| Berkas | Peran |
|-|-|
| `api/intelligence_views.py` | endpoint HTTP engine, kunci API, batas laju |
| `api/intelligence_urls.py` | routing endpoint tersebut |
| `api/openapi.py` | spesifikasi dan panduan untuk konsumen luar |
| `api/docs_views.py` | halaman dokumentasi |
| `api/landing_views.py` | halaman muka |

Model berikut juga milik domain engine, bukan Healthify: `JournalArticle`,
`ConversationSession`, `ConversationMessage`, `ConsultationSummary`,
`ApiAccessRequest`, dan `IntelligenceApiKey`. Memindahkannya berarti migrasi
rename tabel di atas data produksi, jadi paling aman dikerjakan sekali jalan
saat pemisahan, bukan sebelumnya.

Uji engine masih bercampur di `api/tests_intelligence.py` bersama uji lapisan
HTTP Healthify.

Yang sudah dijamin: `EngineBoundaryTests` menggagalkan build bila ada impor ke
aplikasi induk, `EnginePortabilityTests` menjalankan engine dengan model yang
bukan milik Healthify, dan `ragai.checks` melaporkan pemasangan yang kurang
lengkap saat start.

## Isi

| Bagian | Peran |
|-|-|
| `engine.py` | alur satu permintaan dari awal sampai jawaban |
| `contracts.py` | bentuk permintaan, jawaban, dan bukti |
| `runtime.py` | titik sambung ke aplikasi induk |
| `query_understanding/` | klasifikasi intent |
| `context/` | konteks kesehatan dan ingatan percakapan |
| `retrieval/` | pencarian jurnal, leksikon, pelengkapan otomatis |
| `evidence/` | validasi tautan, penilaian mutu, seleksi bukti |
| `reasoning/` | penyusunan jawaban dan pemanggilan LLM |
| `safety/` | pemeriksaan sebelum jawaban dikirim |
| `summarization/` | ringkasan konsultasi |
| `adapters/` | bentuk respons untuk konsumen luar |
