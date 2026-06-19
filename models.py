"""
models.py
---------
Definisi Pydantic models untuk structured output dari LLM.
LangChain menggunakan model ini untuk mem-parse response AI secara terstruktur.

Setiap model digunakan sebagai schema dalam:
- chain.with_structured_output(Model)
"""

from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# MODEL: PERTANYAAN INTERVIEW
# Digunakan oleh: generate_question node di LangGraph
# LangSmith akan menampilkan output terstruktur ini di trace
# ============================================================
class InterviewQuestion(BaseModel):
    """
    Structured output untuk pertanyaan wawancara yang dihasilkan AI.
    """
    question: str = Field(
        description="Pertanyaan wawancara yang akan diajukan kepada kandidat"
    )
    question_type: str = Field(
        description="Jenis pertanyaan: behavioral, technical, situational, atau opening"
    )
    reasoning: str = Field(
        description="Alasan mengapa pertanyaan ini diajukan pada tahap ini"
    )


# ============================================================
# MODEL: EVALUASI JAWABAN
# Digunakan oleh: evaluate_answer node di LangGraph
# LangSmith akan menampilkan detail evaluasi ini di trace panel
# ============================================================
class AnswerEvaluation(BaseModel):
    """
    Structured output untuk evaluasi jawaban kandidat.
    Skor 0-100, beserta kelebihan, kekurangan, dan saran perbaikan.
    """
    score: int = Field(
        ge=0,
        le=100,
        description="Skor jawaban kandidat dari 0 hingga 100"
    )
    strengths: list[str] = Field(
        description="Daftar kelebihan dari jawaban kandidat (minimal 2 poin)"
    )
    weaknesses: list[str] = Field(
        description="Daftar kekurangan dari jawaban kandidat (minimal 1 poin)"
    )
    suggestions: list[str] = Field(
        description="Daftar saran konkret untuk meningkatkan kualitas jawaban"
    )
    brief_feedback: str = Field(
        description="Umpan balik singkat (1-2 kalimat) sebagai transisi ke pertanyaan berikutnya"
    )


# ============================================================
# MODEL: SOFT SKILL SCORES
# Digunakan di dalam FinalReport
# ============================================================
class SoftSkillScores(BaseModel):
    """
    Skor soft skill kandidat berdasarkan keseluruhan sesi wawancara.
    """
    communication: int = Field(
        ge=0,
        le=100,
        description="Skor kemampuan komunikasi kandidat (0-100)"
    )
    problem_solving: int = Field(
        ge=0,
        le=100,
        description="Skor kemampuan pemecahan masalah kandidat (0-100)"
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Skor kepercayaan diri kandidat (0-100)"
    )
    professionalism: int = Field(
        ge=0,
        le=100,
        description="Skor profesionalisme kandidat (0-100)"
    )


# ============================================================
# MODEL: LAPORAN AKHIR
# Digunakan oleh: generate_final_report node di LangGraph
# Merupakan output akhir yang ditampilkan di Streamlit UI
# ============================================================
class FinalReport(BaseModel):
    """
    Structured output untuk laporan akhir sesi wawancara.
    Berisi ringkasan lengkap performa kandidat.
    """
    average_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Skor rata-rata dari seluruh jawaban (0-100)"
    )
    candidate_category: str = Field(
        description="Kategori kandidat: Sangat Baik, Baik, Cukup, atau Perlu Latihan"
    )
    soft_skill_scores: SoftSkillScores = Field(
        description="Skor soft skill kandidat berdasarkan performa keseluruhan"
    )
    candidate_strengths: list[str] = Field(
        description="Daftar kelebihan utama kandidat (minimal 3 poin)"
    )
    improvement_areas: list[str] = Field(
        description="Daftar area yang perlu ditingkatkan (minimal 3 poin)"
    )
    hr_recommendation: str = Field(
        description=(
            "Paragraf rekomendasi dari perspektif HR profesional mengenai "
            "kesiapan kandidat dan langkah selanjutnya"
        )
    )
    overall_summary: str = Field(
        description="Ringkasan singkat performa keseluruhan kandidat (2-3 kalimat)"
    )


# ============================================================
# MODEL: SESI INTERVIEW (untuk serialisasi state)
# ============================================================
class QAPair(BaseModel):
    """
    Pasangan pertanyaan dan jawaban dalam sesi interview.
    """
    question_number: int = Field(description="Nomor urut pertanyaan (1-5)")
    question: str = Field(description="Teks pertanyaan yang diajukan AI")
    answer: str = Field(description="Jawaban yang diberikan kandidat")
    evaluation: Optional[AnswerEvaluation] = Field(
        default=None,
        description="Evaluasi jawaban oleh AI"
    )
