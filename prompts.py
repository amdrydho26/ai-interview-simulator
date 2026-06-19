"""
prompts.py
----------
Kumpulan LangChain PromptTemplate untuk AI Interview Simulator.

Berisi tiga prompt utama:
1. INTERVIEW_PROMPT     — untuk menghasilkan pertanyaan wawancara
2. EVALUATION_PROMPT    — untuk mengevaluasi jawaban kandidat
3. FINAL_REPORT_PROMPT  — untuk membuat laporan akhir wawancara

Setiap prompt menggunakan ChatPromptTemplate dari LangChain,
yang akan ter-trace di LangSmith dengan label sesuai nama prompt.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage


# ============================================================
# 1. INTERVIEW QUESTION PROMPT
# Digunakan di: graph.py -> node generate_question
# LangSmith trace: akan muncul sebagai "ChatPromptTemplate" run
#   dengan input variables: job_role, question_count, conversation_history
# ============================================================
INTERVIEW_SYSTEM_PROMPT = """Anda adalah seorang HR Manager profesional dan berpengalaman dari perusahaan teknologi terkemuka. \
Anda sedang melakukan wawancara kerja untuk posisi {job_role}.

Tugas Anda:
- Mengajukan pertanyaan wawancara yang relevan, realistis, dan terstruktur
- Pertanyaan harus disesuaikan dengan posisi {job_role} dan konteks jawaban sebelumnya
- Gunakan gaya komunikasi yang profesional namun ramah
- Setiap pertanyaan harus menggali kompetensi kandidat secara mendalam

Panduan pertanyaan berdasarkan urutan:
- Pertanyaan 1: Perkenalan diri dan latar belakang (opening)
- Pertanyaan 2: Pengalaman dan keahlian teknis yang relevan (technical)
- Pertanyaan 3: Situasi nyata / studi kasus (situational/behavioral)
- Pertanyaan 4: Motivasi dan pemahaman industri (behavioral)
- Pertanyaan 5: Pertanyaan penutup tentang goals atau ekspektasi (closing)

Ini adalah pertanyaan ke-{question_count} dari 5 pertanyaan total.
Buat pertanyaan yang natural dan mengalir berdasarkan konteks percakapan sebelumnya.
Jangan beri penilaian, cukup ajukan pertanyaan yang tepat."""

INTERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INTERVIEW_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="conversation_history"),
    ("human", "Berikan pertanyaan wawancara berikutnya."),
])


# ============================================================
# 2. EVALUATION PROMPT
# Digunakan di: graph.py -> node evaluate_answer
# LangSmith trace: akan muncul sebagai "ChatPromptTemplate" run
#   dengan input variables: job_role, question, answer, question_number
# ============================================================
EVALUATION_SYSTEM_PROMPT = """Anda adalah seorang HR Evaluator profesional yang menganalisis kualitas jawaban kandidat wawancara kerja.

Posisi yang dilamar: {job_role}
Nomor pertanyaan: {question_number} dari 5

Pertanyaan yang diajukan:
"{question}"

Jawaban kandidat:
"{answer}"

Evaluasi jawaban di atas berdasarkan kriteria berikut:
1. **Relevansi** — Apakah jawaban menjawab pertanyaan dengan tepat?
2. **Kedalaman** — Apakah jawaban cukup detail dan substantif?
3. **Struktur** — Apakah jawaban terorganisir dengan baik?
4. **Profesionalisme** — Apakah jawaban mencerminkan profesionalisme?
5. **Kesesuaian Posisi** — Apakah jawaban sesuai dengan kebutuhan posisi {job_role}?

Berikan evaluasi yang konstruktif, jujur, dan membangun.
Skor harus realistis (jangan terlalu tinggi atau terlalu rendah).
brief_feedback harus positif namun jujur, sebagai transisi ke pertanyaan berikutnya."""

EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EVALUATION_SYSTEM_PROMPT),
    ("human", "Evaluasi jawaban kandidat tersebut dengan detail dan konstruktif."),
])


# ============================================================
# 3. FINAL REPORT PROMPT
# Digunakan di: graph.py -> node generate_final_report
# LangSmith trace: akan muncul sebagai "ChatPromptTemplate" run
#   dengan input variables: job_role, qa_summary, scores_summary
# ============================================================
FINAL_REPORT_SYSTEM_PROMPT = """Anda adalah seorang Senior HR Manager yang membuat laporan evaluasi komprehensif setelah sesi wawancara selesai.

Posisi yang dilamar: {job_role}

Berikut adalah ringkasan sesi wawancara yang telah berlangsung:

{qa_summary}

Ringkasan skor per jawaban:
{scores_summary}

Tugas Anda:
- Buat laporan akhir yang komprehensif dan profesional
- Hitung average_score dari seluruh skor jawaban
- Tentukan candidate_category berdasarkan average_score:
  * 90-100 = Sangat Baik
  * 80-89  = Baik
  * 70-79  = Cukup
  * <70    = Perlu Latihan
- Evaluasi soft skill berdasarkan performa keseluruhan:
  * communication: kemampuan menyampaikan ide dengan jelas
  * problem_solving: kemampuan menganalisis dan memecahkan masalah
  * confidence: tingkat kepercayaan diri dalam menjawab
  * professionalism: sikap dan etika profesional
- Identifikasi minimal 3 kelebihan utama kandidat
- Identifikasi minimal 3 area yang perlu ditingkatkan
- Tulis rekomendasi HR yang profesional dan membangun
- Tulis ringkasan keseluruhan performa kandidat

Gunakan bahasa Indonesia yang profesional dan membangun.
Laporan ini akan diterima langsung oleh kandidat, jadi buatlah motivatif namun jujur."""

FINAL_REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", FINAL_REPORT_SYSTEM_PROMPT),
    ("human", "Buatkan laporan akhir evaluasi wawancara yang komprehensif."),
])


def format_qa_summary(answers: list[dict]) -> str:
    """
    Memformat daftar Q&A menjadi string ringkasan untuk prompt.
    
    Args:
        answers: List dict berisi question, answer, evaluation
    
    Returns:
        String terformat untuk dimasukkan ke prompt
    """
    summary_parts = []
    for i, qa in enumerate(answers, 1):
        part = f"""
--- Pertanyaan {i} ---
Pertanyaan: {qa.get('question', 'N/A')}
Jawaban Kandidat: {qa.get('answer', 'N/A')}
Skor: {qa.get('evaluation', {}).get('score', 'N/A')}/100
"""
        summary_parts.append(part)
    return "\n".join(summary_parts)


def format_scores_summary(answers: list[dict]) -> str:
    """
    Memformat ringkasan skor untuk prompt laporan akhir.
    
    Args:
        answers: List dict berisi evaluation dengan score
    
    Returns:
        String ringkasan skor
    """
    scores = []
    total = 0
    for i, qa in enumerate(answers, 1):
        score = qa.get('evaluation', {}).get('score', 0)
        total += score
        scores.append(f"Pertanyaan {i}: {score}/100")
    
    avg = total / len(answers) if answers else 0
    scores.append(f"\nRata-rata: {avg:.1f}/100")
    return "\n".join(scores)
