"""
Pemeriksaan pemasangan engine.

Dijalankan host lewat kerangka `django.core.checks`, sehingga pemasangan yang
kurang lengkap ketahuan saat start, bukan saat permintaan pertama datang dari
pengguna sungguhan.
"""

from typing import List


def check_installation(app_configs=None, **kwargs) -> List:
    """Laporkan peran yang belum didaftarkan aplikasi induk."""
    from django.core.checks import Error, Warning as CheckWarning

    from . import runtime

    masalah = []

    kurang = [name for name in runtime.REQUIRED_MODELS
              if name not in runtime._models]
    if kurang:
        masalah.append(Error(
            "Model yang dibutuhkan ragai belum didaftarkan: " + ", ".join(kurang),
            hint="Panggil ragai.runtime.configure(models={...}) di AppConfig.ready().",
            id="ragai.E001",
        ))

    if not runtime.service("translate"):
        masalah.append(CheckWarning(
            "Layanan 'translate' tidak terdaftar.",
            hint="Tanpa ini kosakata di luar leksikon tidak terjangkau; "
                 "pertanyaan tentang penyakit yang belum tercatat akan dijawab "
                 "tidak ada bukti.",
            id="ragai.W001",
        ))

    if not runtime.service("embed_article"):
        masalah.append(CheckWarning(
            "Layanan 'embed_article' tidak terdaftar.",
            hint="Jurnal baru hanya akan ditemukan lewat kecocokan kata kunci.",
            id="ragai.W002",
        ))

    return masalah
