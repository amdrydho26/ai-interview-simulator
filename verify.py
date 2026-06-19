"""
verify.py
---------
Script verifikasi semua modul proyek AI Interview Simulator.
Jalankan: py verify.py
"""
import sys

print(f"Python: {sys.version}")
print("=" * 50)

errors = []

# ---- Library Imports ----
print("\n[1] Verifikasi Library Dependencies...")
try:
    from langchain_core.prompts import ChatPromptTemplate
    print("  [OK] langchain-core")
except ImportError as e:
    errors.append(f"langchain-core: {e}")
    print(f"  [FAIL] langchain-core: {e}")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("  [OK] langchain-google-genai")
except ImportError as e:
    errors.append(f"langchain-google-genai: {e}")
    print(f"  [FAIL] langchain-google-genai: {e}")

try:
    from langgraph.graph import StateGraph, START, END
    print("  [OK] langgraph")
except ImportError as e:
    errors.append(f"langgraph: {e}")
    print(f"  [FAIL] langgraph: {e}")

try:
    import langsmith
    print("  [OK] langsmith")
except ImportError as e:
    errors.append(f"langsmith: {e}")
    print(f"  [FAIL] langsmith: {e}")

try:
    from pydantic import BaseModel
    print("  [OK] pydantic")
except ImportError as e:
    errors.append(f"pydantic: {e}")
    print(f"  [FAIL] pydantic: {e}")

try:
    import streamlit
    print("  [OK] streamlit")
except ImportError as e:
    errors.append(f"streamlit: {e}")
    print(f"  [FAIL] streamlit: {e}")

try:
    import dotenv
    print("  [OK] python-dotenv")
except ImportError as e:
    errors.append(f"python-dotenv: {e}")
    print(f"  [FAIL] python-dotenv: {e}")

# ---- Project Module Imports ----
print("\n[2] Verifikasi Modul Proyek...")

try:
    from models import InterviewQuestion, AnswerEvaluation, FinalReport, SoftSkillScores, QAPair
    print("  [OK] models.py - semua Pydantic models loaded")
except Exception as e:
    errors.append(f"models.py: {e}")
    print(f"  [FAIL] models.py: {e}")

try:
    from config import JOB_ROLES, MAX_QUESTIONS, get_candidate_category, validate_config
    print(f"  [OK] config.py - {len(JOB_ROLES)} posisi, max {MAX_QUESTIONS} pertanyaan")
except Exception as e:
    errors.append(f"config.py: {e}")
    print(f"  [FAIL] config.py: {e}")

try:
    from prompts import INTERVIEW_PROMPT, EVALUATION_PROMPT, FINAL_REPORT_PROMPT
    print("  [OK] prompts.py - 3 ChatPromptTemplate loaded")
except Exception as e:
    errors.append(f"prompts.py: {e}")
    print(f"  [FAIL] prompts.py: {e}")

try:
    from graph import InterviewState, build_interview_graph, initialize_state
    print("  [OK] graph.py - LangGraph StateGraph loaded")
except Exception as e:
    errors.append(f"graph.py: {e}")
    print(f"  [FAIL] graph.py: {e}")

# ---- Functional Tests ----
print("\n[3] Verifikasi Fungsional...")

try:
    graph = build_interview_graph()
    print(f"  [OK] LangGraph compiled: {type(graph).__name__}")
except Exception as e:
    errors.append(f"build_interview_graph: {e}")
    print(f"  [FAIL] build_interview_graph: {e}")

try:
    state = initialize_state("Data Scientist")
    assert state["job_role"] == "Data Scientist"
    assert state["question_count"] == 0
    assert state["is_complete"] == False
    print("  [OK] InterviewState initialized correctly")
except Exception as e:
    errors.append(f"initialize_state: {e}")
    print(f"  [FAIL] initialize_state: {e}")

try:
    from config import get_candidate_category
    assert get_candidate_category(95) == "Sangat Baik"
    assert get_candidate_category(85) == "Baik"
    assert get_candidate_category(75) == "Cukup"
    assert get_candidate_category(60) == "Perlu Latihan"
    print("  [OK] Candidate category logic benar")
except Exception as e:
    errors.append(f"get_candidate_category: {e}")
    print(f"  [FAIL] get_candidate_category: {e}")

try:
    q = InterviewQuestion(
        question="Perkenalkan diri Anda",
        question_type="opening",
        reasoning="Pertanyaan pembuka standar"
    )
    assert q.question == "Perkenalkan diri Anda"
    print("  [OK] InterviewQuestion Pydantic model OK")
except Exception as e:
    errors.append(f"InterviewQuestion model: {e}")
    print(f"  [FAIL] InterviewQuestion model: {e}")

# ---- Summary ----
print("\n" + "=" * 50)
if not errors:
    print("SEMUA VERIFIKASI BERHASIL! Proyek siap dijalankan.")
    print("\nJalankan aplikasi dengan:")
    print("  py -m streamlit run app.py")
else:
    print(f"GAGAL: {len(errors)} error ditemukan:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
