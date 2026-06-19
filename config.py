"""
config.py
---------
Konfigurasi aplikasi AI Interview Simulator.
Memuat environment variables dan konstanta aplikasi.
"""

import os
from dotenv import load_dotenv

# Memuat environment variables dari file .env
load_dotenv()

# ============================================================
# LANGSMITH CONFIGURATION
# LangSmith akan melakukan tracing pada setiap chain/graph run.
# Pastikan variabel ini di-set di file .env agar tracing aktif.
# Dashboard: https://smith.langchain.com
# ============================================================
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "AI-Interview-Simulator")
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")

# Set environment variables secara eksplisit agar LangSmith mendeteksinya
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# ============================================================
# GOOGLE GEMINI API CONFIGURATION
# ============================================================
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ============================================================
# APPLICATION SETTINGS
# ============================================================
# Model Gemini yang digunakan
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Suhu model (0.0 = deterministik, 1.0 = kreatif)
MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))

# Jumlah maksimum pertanyaan per sesi interview
MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", "5"))

# ============================================================
# JOB ROLES YANG TERSEDIA
# ============================================================
JOB_ROLES: list[str] = [
    "Data Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "UI/UX Designer",
    "Project Manager",
]

# ============================================================
# KATEGORI KANDIDAT BERDASARKAN SKOR
# ============================================================
CANDIDATE_CATEGORIES: dict[str, tuple[int, int]] = {
    "Sangat Baik": (90, 100),
    "Baik": (80, 89),
    "Cukup": (70, 79),
    "Perlu Latihan": (0, 69),
}


def get_candidate_category(score: float) -> str:
    """
    Menentukan kategori kandidat berdasarkan skor rata-rata.
    
    Args:
        score: Skor rata-rata (0-100)
    
    Returns:
        Kategori kandidat sebagai string
    """
    for category, (low, high) in CANDIDATE_CATEGORIES.items():
        if low <= score <= high:
            return category
    return "Perlu Latihan"


def validate_config() -> tuple[bool, list[str]]:
    """
    Memvalidasi konfigurasi aplikasi.
    
    Returns:
        Tuple (is_valid, list_of_errors)
    """
    errors = []

    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY belum di-set di file .env")

    if not LANGCHAIN_API_KEY:
        errors.append(
            "LANGCHAIN_API_KEY belum di-set (LangSmith tracing tidak aktif)"
        )

    return len(errors) == 0, errors
