import os
import sys
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root path
sys.path.insert(0, os.path.dirname(__file__))

from src.database import init_db, get_db, get_agent_logs, log_agent_decision
from src.data_analyzer import analyze_raw_datasets, generate_master_dataset, train_and_evaluate_ml_pipeline
from src.agents import ProjectPilotOrchestrator
from tests.run_tests import run_all_tests

app = FastAPI(
    title="ProjectPilot AI REST API",
    description="Autonomous Multi-Agent Project Planning, Dynamic Task Allocation & Adaptive Rescue REST API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database Schema
init_db()

orchestrator = ProjectPilotOrchestrator()

# Request Pydantic Schemas
class TeamMemberModel(BaseModel):
    name: str
    skills: List[str]

class CreateProjectRequest(BaseModel):
    project_id: str
    title: str
    goal: str
    start_date: str
    deadline: str
    tech_stack: Optional[str] = "Python, FastAPI, React"
    team_members: List[TeamMemberModel]

class ProgressUpdateRequest(BaseModel):
    progress_updates: Dict[str, float]

class RescueApprovalRequest(BaseModel):
    action: str # APPROVE, REJECT, EDIT
    comment: Optional[str] = None

class AssistantPromptRequest(BaseModel):
    prompt: str

class ResearchQueryRequest(BaseModel):
    project_id: Optional[str] = "DEMO_ATTENDANCE"
    query: str

@app.on_event("startup")
def startup_event():
    """Seed default demo project on API startup so data is immediately available to frontend."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM PROJECTS WHERE project_id = 'DEMO_ATTENDANCE'")
        count = cursor.fetchone()[0]
        
    if count == 0:
        orchestrator.run_initial_planning(
            "DEMO_ATTENDANCE",
            "Smart Attendance System",
            "AI Face recognition attendance system with FastAPI backend, React dashboard, and camera stream.",
            datetime.now().strftime("%Y-%m-%d"),
            (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            "Python, FastAPI, OpenCV, React, Pytest",
            [{"name": "Tharani", "skills": ["Frontend", "UI Design"]}, {"name": "Priya", "skills": ["Backend", "AI Model", "API"]}]
        )

@app.get("/")
def root():
    return {
        "status": "ONLINE",
        "service": "ProjectPilot AI REST API",
        "version": "1.0.0",
        "docs_url": "http://localhost:8000/docs"
    }

@app.post("/api/research/query")
def research_query(req: ResearchQueryRequest):
    """SciSpace-style Research Co-Pilot: Query IEEE literature, database recommendations, and code templates."""
    try:
        resources = orchestrator.run_scispace_research_copilot(req.project_id or "DEMO_ATTENDANCE", req.query)
        return {
            "status": "SUCCESS",
            "query": req.query,
            "resources": resources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/create")
def create_project(req: CreateProjectRequest):
    """Planner Agent & Research Agent: Create project, decompose tasks, allocate work, and store RAG resources."""
    try:
        members = [{"name": m.name, "skills": m.skills} for m in req.team_members]
        sched, workloads, achievable, end_date = orchestrator.run_initial_planning(
            req.project_id, req.title, req.goal, req.start_date, req.deadline, req.tech_stack or "", members
        )
        return {
            "status": "SUCCESS",
            "project_id": req.project_id,
            "projected_completion_date": end_date,
            "deadline_achievable": achievable,
            "tasks_count": len(sched),
            "member_workloads": workloads
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}")
def get_project_details(project_id: str):
    """Retrieve full project state, active tasks, team members, and plan version."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM PROJECTS WHERE project_id = ?", (project_id,))
        proj = cursor.fetchone()
        if not proj:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

        cursor.execute("SELECT * FROM TASKS WHERE project_id = ? ORDER BY task_id ASC", (project_id,))
        tasks = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM TEAM_MEMBERS WHERE project_id = ?", (project_id,))
        members = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM APPROVALS WHERE project_id = ? ORDER BY approval_id DESC LIMIT 1", (project_id,))
        appr = cursor.fetchone()

    return {
        "project": dict(proj),
        "tasks": tasks,
        "team_members": members,
        "active_approval": dict(appr) if appr else None
    }

@app.post("/api/projects/{project_id}/progress")
def update_project_progress(project_id: str, req: ProgressUpdateRequest):
    """Reviewer Agent: Update task progress, calculate health metrics, and trigger Rescue Mode if delayed."""
    try:
        eval_result = orchestrator.run_reviewer_agent(project_id, req.progress_updates)
        return {
            "status": "SUCCESS",
            "evaluation": eval_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_id}/assistant-prompt")
def natural_language_assistant(project_id: str, req: AssistantPromptRequest):
    """AI Mentor & Rescue Assistant: Process student text prompt for technical guidance or blocker re-planning."""
    try:
        result = orchestrator.process_natural_language_assistant(project_id, req.prompt)
        return {
            "status": "SUCCESS",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_id}/rescue-approval")
def rescue_approval(project_id: str, req: RescueApprovalRequest):
    """Human-in-the-Loop: Apply user approval/rejection to proposed Project Rescue plan."""
    try:
        orchestrator.handle_rescue_approval(project_id, req.action)
        return {
            "status": "SUCCESS",
            "action_applied": req.action,
            "message": f"Rescue plan action '{req.action}' executed successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}/logs")
def get_project_agent_logs(project_id: str):
    """Retrieve clean agent activity logs (Planner, Researcher, Reviewer actions)."""
    logs = get_agent_logs(project_id)
    return {"logs": logs}

@app.get("/api/projects/{project_id}/rag-resources")
def get_project_rag_resources(project_id: str):
    """Retrieve technical resources from ChromaDB RAG store."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM RESEARCH_RESOURCES WHERE project_id = ?", (project_id,))
        rows = [dict(r) for r in cursor.fetchall()]
    return {"resources": rows}

@app.get("/api/eval/dataset-analysis")
def get_dataset_analysis():
    """Inspect raw datasets and report synthetic data determination."""
    analysis, _, _ = analyze_raw_datasets()
    return analysis

@app.get("/api/eval/ml-metrics")
def get_ml_metrics():
    """Evaluate 80/10/10 ML Random Forest Classifier pipeline."""
    metrics = train_and_evaluate_ml_pipeline()
    return metrics

@app.post("/api/eval/run-tests")
def trigger_self_tests():
    """Run automated system self-test runner."""
    results = run_all_tests()
    return {"test_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_server:app", host="0.0.0.0", port=8000, reload=False)
