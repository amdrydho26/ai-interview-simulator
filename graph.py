"""
graph.py
--------
Implementasi LangGraph Workflow untuk AI Interview Simulator.

Mendefinisikan:
1. InterviewState   — TypedDict sebagai state LangGraph
2. Node functions   — generate_question, evaluate_answer, generate_final_report
3. Conditional edge — check_question_count (routing logic)
4. Graph builder    — build_interview_graph() yang mengkompilasi workflow

WORKFLOW DIAGRAM:
    START
      ↓
    generate_question
      ↓
    [user input via Streamlit — diluar graph]
      ↓
    evaluate_answer
      ↓
    check_question_count
      ├── question_count < 5  →  generate_question (loop)
      └── question_count = 5  →  generate_final_report
                                        ↓
                                       END

LangSmith Tracing:
- Setiap node akan muncul sebagai child run di LangSmith dashboard
- Nama project: AI-Interview-Simulator
- Setiap LLM call di dalam node akan ter-trace dengan input/output
"""

from typing import TypedDict, Annotated, Any
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    MODEL_TEMPERATURE,
    MAX_QUESTIONS,
)
from models import InterviewQuestion, AnswerEvaluation, FinalReport
from prompts import (
    INTERVIEW_PROMPT,
    EVALUATION_PROMPT,
    FINAL_REPORT_PROMPT,
    format_qa_summary,
    format_scores_summary,
)


# ============================================================
# STATE DEFINITION
# InterviewState adalah "memory" dari seluruh workflow LangGraph.
# Setiap node menerima state ini dan mengembalikan update-nya.
# ============================================================
class InterviewState(TypedDict):
    """
    State utama LangGraph untuk sesi wawancara.
    
    Fields:
        job_role         : Posisi pekerjaan yang dipilih kandidat
        question_count   : Jumlah pertanyaan yang sudah diajukan (0-5)
        current_question : Teks pertanyaan yang sedang aktif
        current_question_type: Jenis pertanyaan saat ini
        answers          : List dict {question, answer, evaluation}
        evaluations      : List AnswerEvaluation (evaluasi setiap jawaban)
        final_report     : FinalReport object (diisi setelah interview selesai)
        conversation_history: List pesan untuk konteks LLM
        is_complete      : Flag apakah interview sudah selesai
    """
    job_role: str
    question_count: int
    current_question: str
    current_question_type: str
    answers: list[dict[str, Any]]
    evaluations: list[dict[str, Any]]
    final_report: dict[str, Any]
    conversation_history: list[dict[str, str]]
    is_complete: bool


def _get_llm() -> ChatGoogleGenerativeAI:
    """
    Menginisialisasi LLM Google Gemini.
    
    Returns:
        Instance ChatGoogleGenerativeAI yang siap digunakan.
    
    LangSmith: Setiap pemanggilan LLM ini akan tercatat sebagai
               "ChatGoogleGenerativeAI" run dalam trace hierarchy.
    """
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=MODEL_TEMPERATURE,
    )


# ============================================================
# NODE 1: GENERATE QUESTION
# Menghasilkan pertanyaan wawancara berikutnya menggunakan LLM.
# 
# LangSmith: Node ini akan muncul sebagai "generate_question" span
#            dengan child span dari ChatGoogleGenerativeAI call.
# ============================================================
def generate_question(state: InterviewState) -> dict[str, Any]:
    """
    Node LangGraph: Menghasilkan pertanyaan wawancara menggunakan LangChain chain.
    
    Chain yang digunakan:
        INTERVIEW_PROMPT | llm.with_structured_output(InterviewQuestion)
    
    Args:
        state: State LangGraph saat ini
    
    Returns:
        Update state dengan current_question dan question_count baru
    """
    llm = _get_llm()
    
    # --------------------------------------------------------
    # LANGCHAIN CHAIN: PromptTemplate | LLM | StructuredOutput
    # LangSmith akan menampilkan chain ini dengan input/output
    # yang jelas di setiap trace run
    # --------------------------------------------------------
    question_chain = INTERVIEW_PROMPT | llm.with_structured_output(InterviewQuestion)
    
    # Bangun conversation history dari answers yang ada
    conversation_history = _build_conversation_history(state)
    
    # Invoke chain — ini yang akan ter-trace di LangSmith
    result: InterviewQuestion = question_chain.invoke({
        "job_role": state["job_role"],
        "question_count": state["question_count"] + 1,
        "conversation_history": conversation_history,
    })
    
    return {
        "current_question": result.question,
        "current_question_type": result.question_type,
        "question_count": state["question_count"] + 1,
    }


# ============================================================
# NODE 2: EVALUATE ANSWER
# Mengevaluasi jawaban kandidat secara terstruktur.
#
# LangSmith: Node ini akan muncul sebagai "evaluate_answer" span
#            dengan detail score, strengths, weaknesses di output
# ============================================================
def evaluate_answer(state: InterviewState) -> dict[str, Any]:
    """
    Node LangGraph: Mengevaluasi jawaban terakhir kandidat menggunakan LangChain chain.
    
    Chain yang digunakan:
        EVALUATION_PROMPT | llm.with_structured_output(AnswerEvaluation)
    
    Args:
        state: State LangGraph dengan jawaban terbaru dari kandidat
    
    Returns:
        Update state dengan evaluasi terbaru ditambahkan ke answers
    """
    llm = _get_llm()
    
    # --------------------------------------------------------
    # LANGCHAIN CHAIN: Evaluation Chain
    # LangSmith akan menampilkan skor dan evaluasi detail di trace
    # --------------------------------------------------------
    evaluation_chain = EVALUATION_PROMPT | llm.with_structured_output(AnswerEvaluation)
    
    # Ambil jawaban terbaru (index terakhir di answers sebelum di-update)
    latest_answer = state["answers"][-1] if state["answers"] else {}
    
    # Invoke evaluation chain
    evaluation: AnswerEvaluation = evaluation_chain.invoke({
        "job_role": state["job_role"],
        "question_number": state["question_count"],
        "question": latest_answer.get("question", state["current_question"]),
        "answer": latest_answer.get("answer", ""),
    })
    
    # Update answers dengan evaluasi
    updated_answers = list(state["answers"])
    if updated_answers:
        updated_answers[-1]["evaluation"] = evaluation.model_dump()
    
    # Tambahkan ke list evaluations
    updated_evaluations = list(state.get("evaluations", []))
    updated_evaluations.append(evaluation.model_dump())
    
    return {
        "answers": updated_answers,
        "evaluations": updated_evaluations,
    }


# ============================================================
# NODE 3: GENERATE FINAL REPORT
# Membuat laporan akhir komprehensif setelah 5 pertanyaan selesai.
#
# LangSmith: Node ini akan muncul sebagai "generate_final_report" span
#            dengan output laporan lengkap yang dapat di-inspect
# ============================================================
def generate_final_report(state: InterviewState) -> dict[str, Any]:
    """
    Node LangGraph: Membuat laporan akhir wawancara menggunakan LangChain chain.
    
    Chain yang digunakan:
        FINAL_REPORT_PROMPT | llm.with_structured_output(FinalReport)
    
    Args:
        state: State LangGraph dengan seluruh jawaban dan evaluasi
    
    Returns:
        Update state dengan final_report dan is_complete=True
    """
    llm = _get_llm()
    
    # --------------------------------------------------------
    # LANGCHAIN CHAIN: Final Report Chain
    # LangSmith akan menampilkan laporan lengkap sebagai output
    # termasuk soft skill scores dan rekomendasi HR
    # --------------------------------------------------------
    report_chain = FINAL_REPORT_PROMPT | llm.with_structured_output(FinalReport)
    
    # Format Q&A summary untuk prompt
    qa_summary = format_qa_summary(state["answers"])
    scores_summary = format_scores_summary(state["answers"])
    
    # Invoke final report chain
    report: FinalReport = report_chain.invoke({
        "job_role": state["job_role"],
        "qa_summary": qa_summary,
        "scores_summary": scores_summary,
    })
    
    return {
        "final_report": report.model_dump(),
        "is_complete": True,
    }


# ============================================================
# CONDITIONAL EDGE: CHECK QUESTION COUNT
# Menentukan routing: lanjut generate pertanyaan atau buat laporan.
#
# LangSmith: Conditional edge ini akan terlihat sebagai routing
#            decision di visualisasi graph trace
# ============================================================
def check_question_count(state: InterviewState) -> str:
    """
    Conditional edge LangGraph: Menentukan langkah selanjutnya.
    
    Logic:
        - Jika question_count < MAX_QUESTIONS (5): kembali ke generate_question
        - Jika question_count >= MAX_QUESTIONS (5): lanjut ke generate_final_report
    
    Args:
        state: State LangGraph saat ini
    
    Returns:
        String nama node berikutnya ("generate_question" atau "generate_final_report")
    """
    if state["question_count"] >= MAX_QUESTIONS:
        return "generate_final_report"
    else:
        return "generate_question"


# ============================================================
# HELPER: BUILD CONVERSATION HISTORY
# ============================================================
def _build_conversation_history(state: InterviewState) -> list:
    """
    Membangun conversation history dari state untuk konteks LLM.
    
    Args:
        state: State LangGraph saat ini
    
    Returns:
        List LangChain messages (HumanMessage, AIMessage)
    """
    messages = []
    for qa in state.get("answers", []):
        if qa.get("question"):
            messages.append(AIMessage(content=qa["question"]))
        if qa.get("answer"):
            messages.append(HumanMessage(content=qa["answer"]))
    return messages


# ============================================================
# GRAPH BUILDER
# Mengkompilasi LangGraph workflow menjadi runnable graph.
#
# LangSmith: Graph ini akan ter-trace sebagai satu "run" dengan
#            semua node sebagai child spans yang dapat di-inspect
#            di https://smith.langchain.com
# ============================================================
def build_interview_graph():
    """
    Membangun dan mengkompilasi LangGraph interview workflow.
    
    Workflow:
        START -> generate_question -> [user input] -> evaluate_answer
              -> check_question_count
              -> (< 5) generate_question [loop]
              -> (= 5) generate_final_report -> END
    
    Returns:
        Compiled LangGraph yang siap di-invoke
    """
    # Inisialisasi StateGraph dengan InterviewState schema
    builder = StateGraph(InterviewState)
    
    # --------------------------------------------------------
    # TAMBAHKAN NODES ke graph
    # Setiap node adalah function yang menerima state dan return update
    # --------------------------------------------------------
    builder.add_node("generate_question", generate_question)
    builder.add_node("evaluate_answer", evaluate_answer)
    builder.add_node("generate_final_report", generate_final_report)
    
    # --------------------------------------------------------
    # TAMBAHKAN EDGES (alur antar node)
    # --------------------------------------------------------
    # Entry point: START -> generate_question
    builder.add_edge(START, "generate_question")
    
    # Setelah evaluate_answer, cek jumlah pertanyaan
    builder.add_conditional_edges(
        source="evaluate_answer",
        path=check_question_count,
        path_map={
            "generate_question": "generate_question",
            "generate_final_report": "generate_final_report",
        }
    )
    
    # generate_final_report -> END
    builder.add_edge("generate_final_report", END)
    
    # --------------------------------------------------------
    # COMPILE GRAPH
    # Mengkompilasi graph menjadi runnable object
    # --------------------------------------------------------
    compiled_graph = builder.compile()
    
    return compiled_graph


# ============================================================
# HELPER FUNCTIONS UNTUK STREAMLIT
# Fungsi-fungsi ini digunakan oleh app.py untuk berinteraksi
# dengan graph secara step-by-step (tidak satu kali invoke)
# ============================================================

def initialize_state(job_role: str) -> InterviewState:
    """
    Menginisialisasi state awal untuk sesi interview baru.
    
    Args:
        job_role: Posisi pekerjaan yang dipilih kandidat
    
    Returns:
        InterviewState yang sudah diinisialisasi
    """
    return InterviewState(
        job_role=job_role,
        question_count=0,
        current_question="",
        current_question_type="opening",
        answers=[],
        evaluations=[],
        final_report={},
        conversation_history=[],
        is_complete=False,
    )


def run_generate_question(state: InterviewState) -> InterviewState:
    """
    Menjalankan node generate_question dan mengupdate state.
    
    Args:
        state: State saat ini
    
    Returns:
        State yang sudah diupdate dengan pertanyaan baru
    """
    update = generate_question(state)
    return {**state, **update}


def run_evaluate_and_check(
    state: InterviewState,
    question: str,
    answer: str
) -> InterviewState:
    """
    Menambahkan jawaban, menjalankan evaluasi, dan mengecek kondisi.
    
    Args:
        state: State saat ini
        question: Pertanyaan yang diajukan
        answer: Jawaban kandidat
    
    Returns:
        State yang sudah diupdate dengan evaluasi dan status terbaru
    """
    # Tambahkan Q&A ke answers
    updated_answers = list(state["answers"])
    updated_answers.append({
        "question": question,
        "answer": answer,
        "evaluation": {},
    })
    
    updated_state = {**state, "answers": updated_answers}
    
    # Jalankan evaluasi
    eval_update = evaluate_answer(updated_state)
    updated_state = {**updated_state, **eval_update}
    
    # Cek apakah interview sudah selesai
    if updated_state["question_count"] >= MAX_QUESTIONS:
        # Generate final report
        report_update = generate_final_report(updated_state)
        updated_state = {**updated_state, **report_update}
    
    return updated_state
