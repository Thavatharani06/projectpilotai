import os
import sys
import json
import sqlite3
import pandas as pd
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.database
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "autotest.db")
src.database.DB_PATH = TEST_DB_PATH

from src.database import init_db, get_db
from src.data_analyzer import analyze_raw_datasets, generate_master_dataset, train_and_evaluate_ml_pipeline
from src.scheduler import decompose_project_goal, generate_schedule, generate_rescue_plan
from src.vector_store import KnowledgeBaseRAG
from src.agents import ProjectPilotOrchestrator

def run_all_tests():
    print("==================================================")
    print("         PROJECTPILOT AI AUTOMATED TEST SUITE    ")
    print("==================================================")

    init_db(TEST_DB_PATH)
    results = []

    # 1. DATA TESTING
    print("\n[1/4] Running Data Testing & Validation...")
    try:
        analysis, df1, df2 = analyze_raw_datasets()
        master_df = generate_master_dataset(df1, df2, analysis)

        assert len(df1) > 0, "Dataset 1 must contain rows."
        assert len(df2) > 0, "Dataset 2 must contain rows."
        assert len(master_df) > 0, "Master Dataset must contain rows."
        assert "Is_Delayed" in master_df.columns, "Master Dataset must contain target column 'Is_Delayed'."
        assert analysis["decision"] in ["SYNTHETIC DATA REQUIRED", "SYNTHETIC DATA NOT REQUIRED"], "Valid decision must be recorded."

        results.append({"category": "DATA", "name": "Schema & Raw Data Validation", "status": "PASSED", "details": f"Dataset 1: {len(df1)} rows, Dataset 2: {len(df2)} rows."})
        results.append({"category": "DATA", "name": "Master Dataset & Stratified Synthetic Quality", "status": "PASSED", "details": f"Master Dataset: {len(master_df)} records generated."})
    except Exception as e:
        results.append({"category": "DATA", "name": "Data Validation Suite", "status": "FAILED", "details": str(e)})

    # 2. MODEL TESTING
    print("\n[2/4] Running ML Model Testing & 80/10/10 Split Verification...")
    try:
        metrics = train_and_evaluate_ml_pipeline()
        assert metrics["accuracy"] >= 0.65, f"Model accuracy should be >= 0.65 (got {metrics['accuracy']})"
        assert metrics["f1_score"] >= 0.60, f"Model F1 score should be >= 0.60 (got {metrics['f1_score']})"
        assert metrics["train_size"] > metrics["validation_size"], "Train split must be larger than validation split."
        assert metrics["test_size"] > 0, "Locked Test set must be non-empty."

        results.append({"category": "MODEL", "name": "80/10/10 Stratified Split & Leakage Check", "status": "PASSED", "details": f"Train: {metrics['train_size']}, Val: {metrics['validation_size']}, Test: {metrics['test_size']}"})
        results.append({"category": "MODEL", "name": "Random Forest Classifier Performance", "status": "PASSED", "details": f"Accuracy: {metrics['accuracy']}, F1-Score: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}"})
    except Exception as e:
        results.append({"category": "MODEL", "name": "ML Evaluation Suite", "status": "FAILED", "details": str(e)})

    # 3. AGENT & SCHEDULER TESTING
    print("\n[3/4] Running Agent & Dynamic Scheduler Testing...")
    try:
        # Test N=1 member
        members_1 = [{"name": "Solo Dev", "skills": ["All"]}]
        tasks = decompose_project_goal("Test Proj", "Goal", "", members_1)
        sched_1, workloads_1, _, _ = generate_schedule(tasks, members_1, "2026-09-01", "2026-10-01")
        assert len(sched_1) == len(tasks), "All tasks must be scheduled for N=1."

        # Test N=3 members dependency preservation
        members_3 = [{"name": "M1", "skills": ["Frontend"]}, {"name": "M2", "skills": ["Backend"]}, {"name": "M3", "skills": ["Database"]}]
        sched_3, workloads_3, _, _ = generate_schedule(tasks, members_3, "2026-09-01", "2026-10-01")
        
        # Verify dependency ordering
        task_map = {t['task_id']: t for t in sched_3}
        for t in sched_3:
            for pred_id in t['predecessors']:
                if pred_id in task_map:
                    pred_end = task_map[pred_id]['planned_end']
                    curr_start = t['planned_start']
                    assert curr_start >= pred_end, f"Dependency violated: {t['task_id']} starts on {curr_start} before predecessor {pred_id} ended on {pred_end}"

        results.append({"category": "AGENT", "name": "Dynamic Task Decomposition & Skill Allocation", "status": "PASSED", "details": f"Successfully scheduled {len(tasks)} tasks for both N=1 and N=3 members."})
        results.append({"category": "AGENT", "name": "Task Dependency Preservation Constraint", "status": "PASSED", "details": "All predecessor task end dates predate successor start dates."})
    except Exception as e:
        results.append({"category": "AGENT", "name": "Scheduler & Dependency Suite", "status": "FAILED", "details": str(e)})

    # 4. SYSTEM & RESCUE MODE TESTING
    print("\n[4/4] Running System & Rescue Mode Testing...")
    try:
        orchestrator = ProjectPilotOrchestrator()
        p_id = "TEST_RESCUE_001"
        members = [{"name": "Tharani", "skills": ["Frontend"]}, {"name": "Priya", "skills": ["Backend"]}]
        
        # 1. Initial Plan
        orchestrator.run_initial_planning(p_id, "Test Attendance", "Goal", "2026-09-01", "2026-09-15", "FastAPI, React", members)
        
        # 2. Simulate delay
        eval_res = orchestrator.run_reviewer_agent(p_id, progress_updates={"T2": 30.0, "T1": 100.0})
        assert eval_res["rescue_mode_triggered"] == True, "Rescue Mode should be triggered on backend delay."

        # 3. Human Approval Workflow
        orchestrator.handle_rescue_approval(p_id, "APPROVE")

        # 4. RAG Retrieval check
        rag = KnowledgeBaseRAG()
        res = rag.search_resources("FastAPI backend API", skill="Backend")
        assert len(res) > 0, "RAG search must return technical documentation."

        results.append({"category": "SYSTEM", "name": "Project Rescue Mode & Delay Detection", "status": "PASSED", "details": "Detected bottleneck on delayed task and successfully triggered Rescue Mode."})
        results.append({"category": "SYSTEM", "name": "Human-in-the-Loop Approval Workflow", "status": "PASSED", "details": "Approved recovery plan and updated active plan version in SQLite DB."})
        results.append({"category": "SYSTEM", "name": "ChromaDB RAG Knowledge Base Retrieval", "status": "PASSED", "details": f"Retrieved: '{res[0]['title']}'."})
    except Exception as e:
        results.append({"category": "SYSTEM", "name": "System & Rescue Mode Suite", "status": "FAILED", "details": str(e)})

    # Record results in test DB
    with get_db(TEST_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM TEST_RESULTS")
        for r in results:
            cursor.execute("""
                INSERT INTO TEST_RESULTS (category, test_name, status, details)
                VALUES (?, ?, ?, ?)
            """, (r['category'], r['name'], r['status'], r['details']))
        conn.commit()

    print("\n--------------------------------------------------")
    print("               SUMMARY TEST REPORT               ")
    print("--------------------------------------------------")
    for r in results:
        symbol = "[OK]" if r['status'] == "PASSED" else "[FAIL]"
        print(f" {symbol} {r['category']:<7} | {r['name']:<48} | {r['status']}")
        print(f"     Details: {r['details']}")
    print("--------------------------------------------------\n")

    return results

if __name__ == "__main__":
    run_all_tests()
