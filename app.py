import os
import sys
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add project root path
sys.path.insert(0, os.path.dirname(__file__))

from src.database import init_db, get_db, get_agent_logs
from src.data_analyzer import analyze_raw_datasets, generate_master_dataset, train_and_evaluate_ml_pipeline
from src.agents import ProjectPilotOrchestrator
from tests.run_tests import run_all_tests

# Initialize Database Schema
init_db()

API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="ProjectPilot AI - SciSpace Academic Co-Pilot & Team Manager",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c20 0%, #1a1635 100%);
        color: #e2e8f0;
    }

    .main-header {
        background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.3rem;
        margin-bottom: 0.1rem;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }

    .search-box {
        background: rgba(30, 27, 60, 0.7);
        border: 2px solid #6c5ce7;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }

    .paper-card {
        background: rgba(30, 27, 60, 0.5);
        border-left: 4px solid #00cec9;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(30, 27, 60, 0.6);
        border: 1px solid rgba(108, 92, 231, 0.25);
        border-radius: 12px;
        padding: 1.0rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        text-align: center;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #00cec9;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
    }

    .rescue-card {
        background: rgba(235, 77, 75, 0.12);
        border: 2px solid #eb4d4b;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    .agent-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .pill-planner { background: rgba(108, 92, 231, 0.3); color: #a29bfe; border: 1px solid #6c5ce7; }
    .pill-researcher { background: rgba(0, 206, 201, 0.3); color: #81ecec; border: 1px solid #00cec9; }
    .pill-reviewer { background: rgba(253, 121, 168, 0.3); color: #ff7675; border: 1px solid #fd79a8; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_orchestrator():
    return ProjectPilotOrchestrator()

orchestrator = get_orchestrator()

# Smart Execution Handler (REST API or In-Memory Cloud Fallback)
def handle_project_create(payload):
    try:
        resp = requests.post(f"{API_BASE_URL}/api/projects/create", json=payload, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    
    members = payload["team_members"]
    sched, workloads, achievable, end_date = orchestrator.run_initial_planning(
        payload["project_id"], payload["title"], payload["goal"],
        payload["start_date"], payload["deadline"], payload.get("tech_stack", ""), members
    )
    return {
        "status": "SUCCESS",
        "project_id": payload["project_id"],
        "projected_completion_date": end_date,
        "deadline_achievable": achievable,
        "tasks_count": len(sched),
        "member_workloads": workloads
    }

def handle_get_project_details(project_id):
    try:
        resp = requests.get(f"{API_BASE_URL}/api/projects/{project_id}", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM PROJECTS WHERE project_id = ?", (project_id,))
        proj = cursor.fetchone()
        if not proj:
            return None

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

def handle_update_progress(project_id, updates):
    try:
        resp = requests.post(f"{API_BASE_URL}/api/projects/{project_id}/progress", json={"progress_updates": updates}, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    eval_res = orchestrator.run_reviewer_agent(project_id, updates)
    return {"status": "SUCCESS", "evaluation": eval_res}

def handle_assistant_prompt(project_id, user_prompt):
    try:
        resp = requests.post(f"{API_BASE_URL}/api/projects/{project_id}/assistant-prompt", json={"prompt": user_prompt}, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    res = orchestrator.process_natural_language_assistant(project_id, user_prompt)
    return {"status": "SUCCESS", "result": res}

def handle_rescue_approval(project_id, action):
    try:
        resp = requests.post(f"{API_BASE_URL}/api/projects/{project_id}/rescue-approval", json={"action": action}, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    orchestrator.handle_rescue_approval(project_id, action)
    return {"status": "SUCCESS"}

# Application Header
st.markdown('<div class="main-header">PROJECTPILOT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">SciSpace-Style Academic Research Co-Pilot & Dynamic Team Manager</div>', unsafe_allow_html=True)

# Central SciSpace-Style Research Co-Pilot Query Bar
with st.container():
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.subheader("🔬 SciSpace Academic Research Co-Pilot")
    query_input = st.text_input("Enter research topic, IEEE paper query, or project idea:", "Real-Time Face Recognition & Automated Attendance with FastAPI & React")
    
    col_q1, col_q2 = st.columns([1, 1])
    with col_q1:
        if st.button("🔎 Search IEEE Literature & Technical Architecture", type="primary", use_container_width=True):
            res_docs = orchestrator.run_scispace_research_copilot(st.session_state.get("active_project_id", "DEMO_ATTENDANCE"), query_input)
            st.success(f"Retrieved {len(res_docs)} IEEE research papers, database recommendations, and starter code templates!")
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/6c5ce7/rocket.png", width=64)
    st.title("Project Controls")
    
    st.subheader("1-Click Demo Projects")
    if st.button("🚀 Demo 1: Smart Attendance (2 Members)", use_container_width=True):
        payload = {
            "project_id": "DEMO_ATTENDANCE",
            "title": "Smart Attendance System",
            "goal": "AI Face recognition attendance system with FastAPI backend, React dashboard, and camera stream.",
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "deadline": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            "tech_stack": "Python, FastAPI, OpenCV, React, Pytest",
            "team_members": [
                {"name": "Tharani", "skills": ["Frontend", "UI Design"]},
                {"name": "Priya", "skills": ["Backend", "AI Model", "API"]}
            ]
        }
        res = handle_project_create(payload)
        if res and res.get("status") == "SUCCESS":
            st.session_state["active_project_id"] = "DEMO_ATTENDANCE"
            st.success("Smart Attendance Demo Loaded!")
            st.rerun()

    if st.button("📱 Demo 2: Mobile Health App (3 Members)", use_container_width=True):
        payload = {
            "project_id": "DEMO_HEALTH",
            "title": "Mobile Health & Fitness Tracker",
            "goal": "Cross-platform mobile app for workout tracking, SQLite storage, and Node.js APIs.",
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "deadline": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
            "tech_stack": "Flutter, Node.js, SQLite, Firebase",
            "team_members": [
                {"name": "Alex", "skills": ["Frontend", "Mobile UI"]},
                {"name": "Bhavya", "skills": ["Backend", "Database"]},
                {"name": "Chris", "skills": ["API Integration", "Testing"]}
            ]
        }
        res = handle_project_create(payload)
        if res and res.get("status") == "SUCCESS":
            st.session_state["active_project_id"] = "DEMO_HEALTH"
            st.success("Mobile Health App Demo Loaded!")
            st.rerun()

# Ensure active project ID in session state
if "active_project_id" not in st.session_state:
    st.session_state["active_project_id"] = "DEMO_ATTENDANCE"

# Auto-plan default demo project if not exists
proj_data = handle_get_project_details(st.session_state["active_project_id"])
if not proj_data:
    payload = {
        "project_id": "DEMO_ATTENDANCE",
        "title": "Smart Attendance System",
        "goal": "AI Face recognition attendance system with FastAPI backend, React dashboard, and camera stream.",
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "deadline": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
        "tech_stack": "Python, FastAPI, OpenCV, React, Pytest",
        "team_members": [
            {"name": "Tharani", "skills": ["Frontend", "UI Design"]},
            {"name": "Priya", "skills": ["Backend", "AI Model", "API"]}
        ]
    }
    handle_project_create(payload)
    proj_data = handle_get_project_details(st.session_state["active_project_id"])

# Navigation Tabs
tab_scispace, tab_team, tab_health, tab_feed = st.tabs([
    "🔬 1. SciSpace Research & Code Specs",
    "👥 2. Dynamic Team Work Allocation",
    "📈 3. Live Health Monitor & Rescue Mode",
    "🤖 4. Agent Decision Feed"
])

# ----------------------------------------------------
# TAB 1: SCISPACE RESEARCH, DATABASE & CODE SPECS
# ----------------------------------------------------
with tab_scispace:
    st.subheader("IEEE Literature Papers, Database Specs & Starter Code")
    active_id = st.session_state["active_project_id"]
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM RESEARCH_RESOURCES WHERE project_id = ?", (active_id,))
        res_rows = [dict(r) for r in cursor.fetchall()]

    if not res_rows:
        st.info("No research papers retrieved yet. Enter a query in the top search bar above!")
    else:
        for r in res_rows:
            st.markdown(f"""
            <div class="paper-card">
                <h4 style="color:#00cec9; margin-top:0;">📄 {r['title']}</h4>
                <p><strong>URL:</strong> <a href="{r['url']}" target="_blank" style="color:#a29bfe;">{r['url']}</a></p>
                <p><strong>Methodology Summary:</strong> {r['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("💡 Database & Code Recommendations")
        rag_docs = orchestrator.rag.search_resources(query_input, top_k=2)
        
        c_code1, c_code2 = st.columns(2)
        with c_code1:
            st.write("### 🗄️ Recommended Database & Storage Specs")
            for d in rag_docs:
                st.info(f"**{d['title']}**\n\n📌 **Recommendation:** {d.get('database_rec', 'SQLite for MVP, PostgreSQL for cloud.')}")
        
        with c_code2:
            st.write("### 💻 Starter Code Snippets")
            for d in rag_docs:
                st.code(d.get('starter_code', '# Starter Code Snippet\nimport sys'), language="python")

# ----------------------------------------------------
# TAB 2: DYNAMIC TEAM WORK ALLOCATION
# ----------------------------------------------------
with tab_team:
    st.subheader("Plan Custom Project & Allocate Team Workload")
    col_l, col_r = st.columns([1, 1])

    with col_l:
        p_code = st.text_input("Project Code", st.session_state.get("active_project_id", "PROJ_001"))
        p_title = st.text_input("Project Title", "Autonomous Crop Disease Drone")
        p_goal = st.text_area("Project Goal & Features", "Drone-based computer vision crop health analyzer using OpenCV, PyTorch, and a Web Dashboard.", height=90)
        p_stack = st.text_input("Technology Stack", "Python, PyTorch, OpenCV, React, FastAPI")

    with col_r:
        c1, c2 = st.columns(2)
        with c1:
            p_start = st.date_input("Start Date", datetime.now().date())
        with c2:
            p_deadline = st.date_input("Deadline", (datetime.now() + timedelta(days=14)).date())

        st.write("**Dynamic Team Setup (Supports Any Team Size)**")
        num_m = st.number_input("Team Size", min_value=1, max_value=15, value=2, step=1)
        
        team_payload = []
        for i in range(int(num_m)):
            default_name = "Tharani" if i == 0 else ("Priya" if i == 1 else f"Member {i+1}")
            default_skills = "Frontend, UI Design" if i == 0 else ("Backend, AI Model" if i == 1 else "Fullstack")
            
            mc1, mc2 = st.columns([1, 1])
            with mc1:
                m_name = st.text_input(f"Member {i+1} Name", value=default_name, key=f"team_mname_{i}")
            with mc2:
                m_skills = st.text_input(f"Member {i+1} Skills", value=default_skills, key=f"team_mskills_{i}")
            team_payload.append({"name": m_name, "skills": [s.strip() for s in m_skills.split(",")]})

    if st.button("✨ Decompose Project & Allocate Schedule", type="primary", use_container_width=True):
        payload = {
            "project_id": p_code,
            "title": p_title,
            "goal": p_goal,
            "start_date": p_start.strftime("%Y-%m-%d"),
            "deadline": p_deadline.strftime("%Y-%m-%d"),
            "tech_stack": p_stack,
            "team_members": team_payload
        }
        res = handle_project_create(payload)
        if res and res.get("status") == "SUCCESS":
            st.session_state["active_project_id"] = p_code
            st.success(f"Project Created & Planned! Projected Completion: {res['projected_completion_date']}")
            st.rerun()

    # Render Project Roadmap & Tasks
    if proj_data:
        p_info = proj_data["project"]
        tasks = proj_data["tasks"]
        
        st.markdown("---")
        st.subheader(f"Calendar Schedule: {p_info['title']} (Plan Version {p_info['active_plan_version']})")
        
        if PLOTLY_AVAILABLE and tasks:
            fig = px.timeline(
                tasks,
                x_start="planned_start",
                x_end="planned_end",
                y="assigned_member_name",
                color="module_name",
                hover_name="task_name",
                text="task_name",
                title="Team Task Allocation & Calendar Schedule",
                labels={"assigned_member_name": "Team Member", "planned_start": "Start Date", "planned_end": "End Date"}
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        if tasks:
            df_t = pd.DataFrame(tasks)
            st.dataframe(
                df_t[["task_name", "module_name", "assigned_member_name", "required_skill", "estimated_days", "planned_start", "planned_end", "actual_progress_pct", "status"]],
                use_container_width=True
            )

# ----------------------------------------------------
# TAB 3: LIVE HEALTH MONITOR & RESCUE MODE
# ----------------------------------------------------
with tab_health:
    if not proj_data or not proj_data.get("tasks"):
        st.info("No active project loaded.")
    else:
        tasks = proj_data["tasks"]
        appr = proj_data.get("active_approval")
        active_id = st.session_state["active_project_id"]

        st.subheader("Student Progress & Live Health Monitor")

        total = len(tasks)
        completed = sum(1 for t in tasks if t['actual_progress_pct'] >= 100.0)
        avg_progress = round(sum(t['actual_progress_pct'] for t in tasks) / total, 1) if total > 0 else 0.0

        cm1, cm2, cm3, cm4 = st.columns(4)
        cm1.markdown(f'<div class="metric-card"><div class="metric-value">{avg_progress}%</div><div class="metric-label">Overall Completion</div></div>', unsafe_allow_html=True)
        cm2.markdown(f'<div class="metric-card"><div class="metric-value">{completed}/{total}</div><div class="metric-label">Tasks Completed</div></div>', unsafe_allow_html=True)
        cm3.markdown(f'<div class="metric-card"><div class="metric-value">{total - completed}</div><div class="metric-label">Tasks In-Progress</div></div>', unsafe_allow_html=True)
        cm4.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00cec9;">HEALTHY</div><div class="metric-label">Project Status</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Update Member Task Progress")
        
        updates = {}
        with st.form("api_progress_form"):
            cols = st.columns(2)
            for idx, t in enumerate(tasks):
                with cols[idx % 2]:
                    updates[t['task_id']] = st.slider(
                        f"📌 {t['task_name']} ({t['assigned_member_name']})",
                        min_value=0.0, max_value=100.0, value=float(t['actual_progress_pct']), step=10.0, key=f"api_prog_{t['task_id']}"
                    )
            btn_save = st.form_submit_button("💾 Save Progress & Evaluate Health", use_container_width=True)

        if btn_save:
            res = handle_update_progress(active_id, updates)
            if res and res.get("status") == "SUCCESS":
                eval_res = res["evaluation"]
                st.success(f"Progress Saved! Reviewer Agent evaluated project health. Risk Level: {eval_res['risk_level']}")
                if eval_res["rescue_mode_triggered"]:
                    st.warning("⚠️ Reviewer Agent detected bottleneck and triggered Project Rescue Mode!")
                st.rerun()

        st.markdown("---")
        st.subheader("🚨 Interactive AI Mentor & Rescue Assistant")
        st.caption("Ask ANY technical question OR report a project delay/blocker in plain English:")

        user_prompt = st.text_input("Enter question or blocker:", "Priya is delayed by 3 days on Backend API Setup due to database connection errors.")
        if st.button("🤖 Ask AI Mentor & Rescue Assistant", type="secondary", use_container_width=True):
            res = handle_assistant_prompt(active_id, user_prompt)
            if res and res.get("status") == "SUCCESS":
                a_res = res["result"]
                if a_res["type"] == "RESCUE_TRIGGERED":
                    st.warning(f"🚨 {a_res['message']}")
                else:
                    st.info(f"💡 {a_res['message']}")
                st.rerun()

        # Render Rescue Approval Panel if active
        if appr and appr.get('status') == 'PENDING':
            st.markdown("""
            <div class="rescue-card">
                <h3 style="color:#eb4d4b; margin-top:0;">⚠️ PROJECT RESCUE PLAN PENDING APPROVAL</h3>
                <p>The Planner Agent analyzed the bottleneck and re-balanced the remaining workload across team members.</p>
            </div>
            """, unsafe_allow_html=True)
            st.write(f"**Proposed Recovery Strategy:** {appr['user_comment']}")

            ca, cb = st.columns(2)
            with ca:
                if st.button("✅ APPROVE & ACTIVATE RECOVERY SCHEDULE", type="primary", use_container_width=True):
                    res = handle_rescue_approval(active_id, "APPROVE")
                    if res:
                        st.success("Recovery plan APPROVED! Project plan version updated.")
                        st.rerun()
            with cb:
                if st.button("❌ REJECT RECOVERY SCHEDULE", use_container_width=True):
                    res = handle_rescue_approval(active_id, "REJECT")
                    if res:
                        st.info("Recovery plan rejected.")
                        st.rerun()

# ----------------------------------------------------
# TAB 4: AGENT DECISION FEED
# ----------------------------------------------------
with tab_feed:
    st.subheader("Agentic Reasoning & Decision Log")
    
    st.info("""
    💡 **What is this tab?**  
    This tab is your **AI Activity Feed & Transparent Decision Log**. It records every action taken behind the scenes by your specialized AI Agents (**Planner Agent**, **Research Agent**, and **Reviewer Agent**).
    """)
    
    active_id = st.session_state["active_project_id"]
    logs = get_agent_logs(active_id)

    if not logs:
        st.info("No agent actions logged for this project yet.")
    else:
        for l in logs:
            pill_class = "pill-planner" if "Planner" in l['agent_name'] else ("pill-researcher" if "Research" in l['agent_name'] else "pill-reviewer")
            st.markdown(f"""
            <div style="background: rgba(30, 27, 60, 0.4); border-left: 4px solid #6c5ce7; padding: 0.8rem; margin-bottom: 0.6rem; border-radius: 4px;">
                <span class="agent-pill {pill_class}">{l['agent_name']}</span>
                <strong style="color:#00cec9;">{l['action']}</strong>
                <span style="float:right; font-size:0.75rem; color:#64748b;">{l['created_at']}</span>
                <p style="margin-top:0.4rem; font-size:0.9rem; color:#cbd5e1;">{l['reasoning']}</p>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------
# BOTTOM EXPANDER: ADVANCED TECHNICAL DIAGNOSTICS
# ----------------------------------------------------
st.markdown("---")
with st.expander("🔬 Advanced System Diagnostics, Dataset Analysis & ML Metrics"):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write("### Dataset Analysis & Decision")
        ds_data, _, _ = analyze_raw_datasets()
        if ds_data:
            st.json(ds_data)

        if st.button("⚡ Evaluate 80/10/10 ML Delay Model", use_container_width=True):
            ml_data = train_and_evaluate_ml_pipeline()
            if ml_data:
                st.json(ml_data)

    with col_d2:
        st.write("### Self-Test Suite Runner")
        if st.button("🧪 Run Self-Tests", use_container_width=True):
            test_res = run_all_tests()
            st.success("Tests Executed!")
            st.json(test_res)
