"""
app.py
------
Streamlit UI untuk AI Interview Simulator.

Halaman utama aplikasi dengan tampilan modern dan profesional.
Menggunakan LangGraph workflow dari graph.py untuk mengelola sesi interview.

Fitur UI:
- Sidebar: Pilih posisi & mulai interview
- Main Area: Pertanyaan aktif + input jawaban
- Laporan Akhir: Metric cards, soft skill scores, kelebihan, kekurangan, rekomendasi
"""

import streamlit as st
import time
from typing import Any

# Import konfigurasi & graph
from config import (
    JOB_ROLES,
    MAX_QUESTIONS,
    validate_config,
    get_candidate_category,
)
from graph import (
    initialize_state,
    run_generate_question,
    run_evaluate_and_check,
    InterviewState,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Interview Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Modern Dark UI
# ============================================================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Reset & Font */
    * { font-family: 'Inter', sans-serif !important; }
    
    /* App Background */
    .stApp {
        background: #171717;
        min-height: 100vh;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* ---- HEADER ---- */
    .hero-header {
        text-align: center;
        padding: 2.5rem 2rem 1.5rem;
        background: #1E1E20;
        border: 2px solid #282828;
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .hero-header h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        background: #6865F2;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 0.5rem !important;
    }
    .hero-header p {
        color: #808082 !important;
        font-size: 1rem !important;
        margin: 0 0 2rem 0 !important;
    }

    /* ---- QUESTION CARD ---- */
    .question-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1));
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .question-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        background: linear-gradient(180deg, #a78bfa, #60a5fa);
        border-radius: 4px 0 0 4px;
    }
    .question-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }
    .question-text {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
        line-height: 1.6;
    }

    /* ---- PROGRESS BAR ---- */
    .progress-container {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.6rem;
        color: #94a3b8;
        font-size: 0.85rem;
    }
    .progress-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        transition: width 0.5s ease;
    }

    /* ---- METRIC CARD ---- */
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.12));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ---- BADGE ---- */
    .badge-excellent { background: linear-gradient(135deg, #34d399, #059669); color: white; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; display: inline-block; }
    .badge-good { background: linear-gradient(135deg, #60a5fa, #2563eb); color: white; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; display: inline-block; }
    .badge-fair { background: linear-gradient(135deg, #fbbf24, #d97706); color: white; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; display: inline-block; }
    .badge-poor { background: linear-gradient(135deg, #f87171, #dc2626); color: white; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; display: inline-block; }

    /* ---- SKILL BAR ---- */
    .skill-item {
        margin-bottom: 1rem;
    }
    .skill-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.4rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .skill-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        height: 10px;
        overflow: hidden;
    }
    .skill-bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    /* ---- LIST ITEMS ---- */
    .list-item-strength {
        background: rgba(52,211,153,0.08);
        border: 1px solid rgba(52,211,153,0.2);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        color: #d1fae5;
        font-size: 0.92rem;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }
    .list-item-improvement {
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.2);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        color: #fef3c7;
        font-size: 0.92rem;
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
    }

    /* ---- RECOMMENDATION BOX ---- */
    .recommendation-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1));
        border: 1px solid rgba(99,102,241,0.35);
        border-radius: 16px;
        padding: 1.5rem 1.8rem;
        color: #e2e8f0;
        line-height: 1.8;
        font-size: 0.95rem;
        position: relative;
    }
    .recommendation-box::before {
        content: '"';
        font-size: 5rem;
        color: rgba(167,139,250,0.2);
        position: absolute;
        top: -1rem;
        left: 1rem;
        font-family: Georgia, serif;
        line-height: 1;
    }

    /* ---- SECTION HEADER ---- */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* ---- EVALUATION CARD ---- */
    .eval-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .eval-score {
        font-size: 1.8rem;
        font-weight: 800;
        color: #a78bfa;
    }

    /* ---- SIDEBAR ---- */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #111111 !important;
        border-right: 1px solid rgba(99,102,241,0.2) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
    }

    /* Textarea */
    .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #a78bfa !important;
        box-shadow: 0 0 0 2px rgba(167,139,250,0.15) !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }

    /* Warning/Success/Info boxes */
    .stAlert {
        border-radius: 12px !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Spinner */
    .stSpinner > div {
        border-color: #a78bfa transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# Streamlit session_state digunakan sebagai persistent storage
# antar re-render. interview_state menyimpan InterviewState LangGraph.
# ============================================================
def init_session_state():
    """Menginisialisasi session state Streamlit."""
    defaults = {
        "interview_state": None,       # InterviewState LangGraph
        "phase": "setup",              # "setup" | "interviewing" | "complete"
        "selected_role": JOB_ROLES[0], # Posisi yang dipilih
        "current_answer": "",          # Jawaban yang sedang ditulis
        "is_loading": False,           # Loading state untuk animasi
        "error_message": "",           # Pesan error
        "question_history": [],        # Riwayat Q&A untuk tampilan
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ============================================================
# HELPER: Render Header
# ============================================================
def render_header():
    """Menampilkan hero header aplikasi."""
    st.markdown("""
    <div class="hero-header">
        <h1>🎯 AI Interview Simulator</h1>
        <p>Sistem Simulasi Wawancara Kerja Berbasis Large Language Model</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HELPER: Render Progress Bar
# ============================================================
def render_progress(current: int, total: int = MAX_QUESTIONS):
    """Menampilkan progress bar sesi interview."""
    pct = int((current / total) * 100)
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-label">
            <span>Progress Wawancara</span>
            <span><strong style="color:#a78bfa">{current}</strong> / {total} Pertanyaan</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: {pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HELPER: Render Question Card
# ============================================================
def render_question_card(question_num: int, question_text: str, q_type: str = ""):
    """Menampilkan kartu pertanyaan dengan styling premium."""
    type_icons = {
        "opening": "", "technical": "",
        "behavioral": "", "situational": "", "closing": "",
    }
    icon = type_icons.get(q_type.lower(), "?")
    type_label = q_type.upper() if q_type else "INTERVIEW"

    st.markdown(f"""
    <div class="question-card">
        <div class="question-label">{icon} PERTANYAAN {question_num} — {type_label}</div>
        <div class="question-text">{question_text}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HELPER: Render Score Badge
# ============================================================
def get_score_badge(score: float) -> str:
    """Mendapatkan HTML badge berdasarkan skor."""
    if score >= 90:
        return f'<span class="badge-excellent">✨ Sangat Baik</span>'
    elif score >= 80:
        return f'<span class="badge-good">👍 Baik</span>'
    elif score >= 70:
        return f'<span class="badge-fair">📈 Cukup</span>'
    else:
        return f'<span class="badge-poor">💪 Perlu Latihan</span>'


# ============================================================
# PHASE 1: SETUP PAGE
# ============================================================
def render_setup_page():
    """Menampilkan halaman setup untuk memilih posisi dan memulai interview."""
    render_header()

    # Validasi konfigurasi
    is_valid, errors = validate_config()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
             border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;">
            <h4 style="color: #e2e8f0; margin-top:0; margin-bottom: 0.8rem;">Cara Menggunakan</h4>
            <div style="color: #808082; font-size: 0.9rem; line-height: 1.2;">
                <p><strong style="color:#C0C0C0">1. Pilih posisi</strong> pekerjaan yang ingin dilamar</p>
                <p><strong style="color:#C0C0C0">2. Klik mulai</strong> untuk memulai sesi interview</p>
                <p><strong style="color:#C0C0C0">3. Jawab pertanyaan</strong> AI dengan teks jawaban Anda</p>
                <p><strong style="color:#C0C0C0">4. Terima evaluasi</strong> setelah 5 pertanyaan selesai</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
             border-radius: 16px; padding: 1.5rem;">
            <h4 style="color: #e2e8f0; margin-top:0; margin-bottom: 0.4rem;">Teknologi</h4>
            <div style="color: #808082; font-size: 0.85rem; line-height: 2;">
                <div><strong style="color:#C0C0C0">LangChain</strong> — Chains & Structured Output</div>
                <div><strong style="color:#C0C0C0">LangGraph</strong> — Stateful Workflow</div>
                <div><strong style="color:#C0C0C0">LangSmith</strong> — Tracing & Monitoring</div>
                <div><strong style="color:#C0C0C0">Google Gemini</strong> — LLM Engine</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #1E1E20; margin-bottom: 1.5rem;
             border: 2px solid #282828; border-radius: 16px; padding: 1.8rem;">
            <h4 style="color: #e2e8f0; margin: 0.7rem 0 0 0">Mulai Interview</h4>
        """, unsafe_allow_html=True)

        selected = st.selectbox(
            "Pilih Posisi Pekerjaan",
            options=JOB_ROLES,
            index=JOB_ROLES.index(st.session_state.selected_role),
            key="role_select",
        )
        st.session_state.selected_role = selected

        st.markdown(f"""
        <div style="background: #1E1E20; border-radius: 10px;
             padding: 0.8rem 1rem; margin: 0.8rem 0; font-size: 0.85rem; color: #808082;">
            Anda akan diwawancarai sebagai calon <strong style="color:#C0C0C0">{selected}</strong>.
            Sesi terdiri dari <strong style="color:#C0C0C0">5 pertanyaan</strong> yang semakin mendalam.
        </div>
        """, unsafe_allow_html=True)

        if not is_valid:
            for err in errors:
                if "GOOGLE_API_KEY" in err:
                    st.error(f"❌ {err}")
                else:
                    st.warning(f"⚠️ {err}")

        # Disable tombol jika GOOGLE_API_KEY tidak ada
        google_key_missing = not any(
            True for e in errors if "GOOGLE_API_KEY" not in e
        ) and any(True for e in errors if "GOOGLE_API_KEY" in e)

        if st.button("Mulai Sesi Interview", disabled=google_key_missing, use_container_width=True):
            start_interview()

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PHASE 2: INTERVIEW PAGE
# ============================================================
def render_interview_page():
    """Menampilkan halaman sesi interview aktif."""
    state: InterviewState = st.session_state.interview_state

    # Header compact
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between;
         background: #1E1E20; border: 1px solid #282828;
         border-radius: 14px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
        <div>
            <span style="color:#C0C0C0; font-weight:700; font-size:1.1rem;">AI Interview Simulator</span>
            <span style="color:#64748b; margin:0 0.8rem;">•</span>
            <span style="color:#94a3b8; font-size:0.9rem;">{state['job_role']}</span>
        </div>
        <div style="color:#94a3b8; font-size:0.85rem;">
            Pertanyaan <strong style="color:#C0C0C0">{state['question_count']}</strong> / {MAX_QUESTIONS}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    render_progress(state["question_count"], MAX_QUESTIONS)

    # Tampilkan riwayat pertanyaan sebelumnya (collapsed)
    if st.session_state.question_history:
        with st.expander(f"📜 Riwayat Pertanyaan ({len(st.session_state.question_history)} sebelumnya)", expanded=False):
            for i, hist in enumerate(st.session_state.question_history, 1):
                eval_data = hist.get("evaluation", {})
                score = eval_data.get("score", "-")
                st.markdown(f"""
                <div class="eval-card">
                    <div style="color:#a78bfa; font-size:0.75rem; font-weight:600; text-transform:uppercase; margin-bottom:0.5rem;">
                        Pertanyaan {i}
                    </div>
                    <div style="color:#e2e8f0; font-size:0.9rem; margin-bottom:0.5rem;">❓ {hist.get('question','')}</div>
                    <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:0.5rem;">💬 {hist.get('answer','')[:150]}{'...' if len(hist.get('answer','')) > 150 else ''}</div>
                    <div style="color:#34d399; font-size:0.85rem; font-weight:600;">📊 Skor: {score}/100</div>
                </div>
                """, unsafe_allow_html=True)

    # Tampilkan pertanyaan aktif
    if state["current_question"]:
        render_question_card(
            state["question_count"],
            state["current_question"],
            state.get("current_question_type", "")
        )

        # Input jawaban
        st.markdown('<div style="margin-bottom:0.5rem; color:#94a3b8; font-size:0.85rem;">Ketik jawaban Anda di bawah ini:</div>', unsafe_allow_html=True)

        answer = st.text_area(
            label="Jawaban Anda",
            placeholder="Tuliskan jawaban Anda secara lengkap dan jelas...\n\nContoh: Saya memiliki pengalaman 3 tahun dalam bidang data analysis, di mana saya...",
            height=180,
            key=f"answer_{state['question_count']}",
            label_visibility="collapsed",
        )

        col_submit, col_tip = st.columns([1, 2])
        with col_submit:
            submit_clicked = st.button(
                "📨 Kirim Jawaban",
                disabled=st.session_state.is_loading,
                use_container_width=True,
            )

        with col_tip:
            st.markdown("""
            <div style="color:#64748b; font-size:0.8rem; padding-top:0.7rem;">
             <em>Jawablah dengan lengkap. AI akan mengevaluasi kualitas, relevansi, dan kedalaman jawaban Anda.</em>
            </div>
            """, unsafe_allow_html=True)

        if submit_clicked:
            if not answer or len(answer.strip()) < 10:
                st.warning("Jawaban terlalu singkat. Harap jawab dengan lebih lengkap (minimal 10 karakter).")
            else:
                process_answer(answer.strip())
                st.rerun()

    # Loading state
    if st.session_state.is_loading:
        with st.spinner("🤖 AI sedang memproses..."):
            time.sleep(0.5)


# ============================================================
# PHASE 3: RESULTS PAGE
# ============================================================
def render_results_page():
    """Menampilkan laporan akhir wawancara dengan visualisasi lengkap."""
    state: InterviewState = st.session_state.interview_state
    report: dict[str, Any] = state.get("final_report", {})

    if not report:
        st.error("❌ Laporan tidak tersedia. Silakan mulai sesi baru.")
        return

    # ---- HERO RESULTS ----
    avg_score = report.get("average_score", 0)
    category = report.get("candidate_category", "N/A")
    badge_html = get_score_badge(avg_score)

    st.markdown(f"""
    <div style="text-align:center; padding: 2.5rem 2rem;
         background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.15));
         border: 1px solid rgba(99,102,241,0.3); border-radius: 20px; margin-bottom: 2rem;">
        <div style="font-size:4rem; margin-bottom:0.5rem;">🎉</div>
        <h2 style="color:#e2e8f0; margin:0 0 0.5rem; font-size:1.8rem; font-weight:800;">
            Interview Selesai!
        </h2>
        <p style="color:#94a3b8; margin-bottom:1rem; font-size:0.95rem;">
            Posisi: <strong style="color:#a78bfa">{state['job_role']}</strong>
        </p>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

    # ---- METRIC CARDS ----
    soft_skills = report.get("soft_skill_scores", {})
    comm_score = soft_skills.get("communication", 0)
    ps_score = soft_skills.get("problem_solving", 0)
    conf_score = soft_skills.get("confidence", 0)
    prof_score = soft_skills.get("professionalism", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    metric_data = [
        (c1, f"{avg_score:.0f}", "Skor Rata-rata"),
        (c2, f"{comm_score}", "Komunikasi"),
        (c3, f"{ps_score}", "Problem Solving"),
        (c4, f"{conf_score}", "Kepercayaan Diri"),
        (c5, f"{prof_score}", "Profesionalisme"),
    ]
    for col, val, label in metric_data:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- OVERALL SUMMARY ----
    overall = report.get("overall_summary", "")
    if overall:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
             border-radius: 14px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; color:#94a3b8; font-size:0.95rem; line-height:1.7;">
            📝 {overall}
        </div>
        """, unsafe_allow_html=True)

    # ---- SOFT SKILLS VISUAL BARS ----
    st.markdown('<div class="section-header">📊 Soft Skill Assessment</div>', unsafe_allow_html=True)

    skills_data = [
        ("💬 Komunikasi", comm_score, "#a78bfa"),
        ("🧩 Problem Solving", ps_score, "#60a5fa"),
        ("💪 Kepercayaan Diri", conf_score, "#34d399"),
        ("👔 Profesionalisme", prof_score, "#fbbf24"),
    ]
    for skill_name, skill_score, color in skills_data:
        st.markdown(f"""
        <div class="skill-item">
            <div class="skill-header">
                <span>{skill_name}</span>
                <span style="color:{color}; font-weight:700;">{skill_score}/100</span>
            </div>
            <div class="skill-bar-bg">
                <div class="skill-bar-fill" style="width:{skill_score}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- STRENGTHS & IMPROVEMENTS ----
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-header">✅ Kelebihan Kandidat</div>', unsafe_allow_html=True)
        strengths = report.get("candidate_strengths", [])
        for strength in strengths:
            st.markdown(f"""
            <div class="list-item-strength">
                <span>✓</span>
                <span>{strength}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-header">🎯 Area Perbaikan</div>', unsafe_allow_html=True)
        improvements = report.get("improvement_areas", [])
        for improvement in improvements:
            st.markdown(f"""
            <div class="list-item-improvement">
                <span>→</span>
                <span>{improvement}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- HR RECOMMENDATION ----
    st.markdown('<div class="section-header">💼 Rekomendasi HR Profesional</div>', unsafe_allow_html=True)
    hr_rec = report.get("hr_recommendation", "")
    st.markdown(f"""
    <div class="recommendation-box">
        <div style="padding-left:2rem;">{hr_rec}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- DETAIL EVALUASI PER PERTANYAAN ----
    st.markdown('<div class="section-header">📋 Detail Evaluasi Per Pertanyaan</div>', unsafe_allow_html=True)

    for i, qa in enumerate(state.get("answers", []), 1):
        eval_data = qa.get("evaluation", {})
        score = eval_data.get("score", 0)
        with st.expander(f"Pertanyaan {i} — Skor: {score}/100", expanded=False):
            st.markdown(f"""
            <div style="color:#a78bfa; font-size:0.85rem; margin-bottom:0.5rem; font-weight:600;">❓ PERTANYAAN</div>
            <div style="color:#e2e8f0; margin-bottom:1rem; line-height:1.6;">{qa.get('question','')}</div>
            <div style="color:#a78bfa; font-size:0.85rem; margin-bottom:0.5rem; font-weight:600;">💬 JAWABAN ANDA</div>
            <div style="color:#94a3b8; margin-bottom:1rem; line-height:1.6;">{qa.get('answer','')}</div>
            """, unsafe_allow_html=True)

            sc1, sc2 = st.columns([1, 3])
            with sc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{score}</div>
                    <div class="metric-label">Skor</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                brief = eval_data.get("brief_feedback", "")
                if brief:
                    st.info(f"💡 {brief}")

            col_s, col_w = st.columns(2)
            with col_s:
                st.markdown("**✅ Kelebihan:**")
                for s in eval_data.get("strengths", []):
                    st.markdown(f"• {s}")
            with col_w:
                st.markdown("**⚠️ Kekurangan:**")
                for w in eval_data.get("weaknesses", []):
                    st.markdown(f"• {w}")

            sugg = eval_data.get("suggestions", [])
            if sugg:
                st.markdown("**💡 Saran Perbaikan:**")
                for sug in sugg:
                    st.markdown(f"→ {sug}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- TOMBOL ULANG ----
    if st.button("Mulai Interview Baru", use_container_width=False):
        reset_session()
        st.rerun()


# ============================================================
# ACTION FUNCTIONS
# ============================================================
def start_interview():
    """Memulai sesi interview baru."""
    job_role = st.session_state.selected_role

    # Inisialisasi state LangGraph
    state = initialize_state(job_role)

    with st.spinner("Menyiapkan sesi interview..."):
        try:
            # Generate pertanyaan pertama
            state = run_generate_question(state)
            st.session_state.interview_state = state
            st.session_state.phase = "interviewing"
            st.session_state.question_history = []
            st.session_state.error_message = ""
        except Exception as e:
            st.session_state.error_message = f"Error: {str(e)}"
            st.error(f"Gagal memulai interview: {str(e)}")


def process_answer(answer: str):
    """Memproses jawaban kandidat dan menjalankan evaluasi."""
    state: InterviewState = st.session_state.interview_state
    current_question = state["current_question"]
    current_q_type = state.get("current_question_type", "")

    with st.spinner("AI sedang mengevaluasi jawaban Anda..."):
        try:
            # Simpan ke history sebelum diproses
            st.session_state.question_history.append({
                "question": current_question,
                "answer": answer,
                "question_type": current_q_type,
            })

            # Jalankan evaluasi dan cek kondisi via LangGraph nodes
            updated_state = run_evaluate_and_check(state, current_question, answer)
            st.session_state.interview_state = updated_state

            # Update history dengan evaluasi
            if updated_state["answers"]:
                last_eval = updated_state["answers"][-1].get("evaluation", {})
                st.session_state.question_history[-1]["evaluation"] = last_eval

            if updated_state.get("is_complete", False):
                st.session_state.phase = "complete"
            else:
                # Generate pertanyaan berikutnya
                next_state = run_generate_question(updated_state)
                st.session_state.interview_state = next_state

        except Exception as e:
            st.error(f"Error saat memproses jawaban: {str(e)}")
            st.session_state.error_message = str(e)


def reset_session():
    """Mereset seluruh session state ke kondisi awal."""
    keys_to_reset = [
        "interview_state", "phase", "current_answer",
        "is_loading", "error_message", "question_history"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    """Menampilkan sidebar dengan informasi dan kontrol."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;">
            <div style="font-size:6rem;">🎯</div>
            <div style="font-weight:800; font-size:1.1rem; color:#e2e8f0;">AI Interview Simulator</div>
            <div style="color:#64748b; font-size:0.75rem; margin-top:0.2rem;">v1.0.0</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        phase = st.session_state.get("phase", "setup")

        if phase == "setup":
            st.markdown("**Status:** `Belum dimulai`")
            st.markdown("""
            <div style="color:#64748b; font-size:0.8rem; line-height:1.7; margin-top:0.5rem;">
                Pilih posisi pekerjaan dan klik <strong>Mulai Interview</strong> untuk memulai sesi wawancara.
            </div>
            """, unsafe_allow_html=True) 

        elif phase == "interviewing":
            state = st.session_state.interview_state
            q_count = state["question_count"] if state else 0
            st.markdown(f"**Status:** `Sesi Aktif`")
            st.markdown(f"**Posisi:** `{state['job_role'] if state else '-'}`")
            st.markdown(f"**Pertanyaan:** `{q_count} / {MAX_QUESTIONS}`")

            st.divider()
            if st.button("Hentikan Sesi", use_container_width=True):
                reset_session()
                st.rerun()

        elif phase == "complete":
            state = st.session_state.interview_state
            report = state.get("final_report", {}) if state else {}
            avg_score = report.get("average_score", 0)

            st.markdown(f"**Status:** `Selesai`")
            st.markdown(f"**Posisi:** `{state['job_role'] if state else '-'}`")
            st.markdown(f"**Skor:** `{avg_score:.1f}/100`")
            st.markdown(f"**Kategori:** `{report.get('candidate_category', '-')}`")

            st.divider()
            if st.button("🔄 Interview Baru", use_container_width=True):
                reset_session()
                st.rerun()

        st.divider()

        # LangSmith status
        from config import LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
        langsmith_active = bool(LANGCHAIN_API_KEY)

        st.markdown(f"""
        <div style="background: rgba({'52,211,153' if langsmith_active else '248,113,113'},0.08);
             border: 1px solid rgba({'52,211,153' if langsmith_active else '248,113,113'},0.25);
             border-radius: 10px; padding: 0.8rem; font-size:0.78rem;">
            <div style="font-weight:600; color:#{'34d399' if langsmith_active else 'f87171'}; margin-bottom:0.3rem;">
                {'🟢' if langsmith_active else '🔴'} LangSmith Tracing
            </div>
            <div style="color:#64748b;">
                {'Aktif — ' + LANGCHAIN_PROJECT if langsmith_active else 'Tidak aktif (API key belum di-set)'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="color:#374151; font-size:0.72rem; text-align:center; margin-top:1.5rem;">
            Powered by LangChain · LangGraph<br>LangSmith · Google Gemini
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# MAIN APP
# ============================================================
def main():
    """Entry point utama aplikasi Streamlit."""
    init_session_state()
    render_sidebar()

    phase = st.session_state.get("phase", "setup")

    if phase == "setup":
        render_setup_page()
    elif phase == "interviewing":
        render_interview_page()
    elif phase == "complete":
        render_results_page()
    else:
        render_setup_page()


if __name__ == "__main__":
    main()
