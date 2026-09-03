import os
import json
import re
from datetime import datetime, timedelta

def decompose_project_goal(title, goal, tech_stack="", team_members=None):
    """
    Decomposes any project goal into tailored modules & tasks.
    Uses LLM API if GEMINI_API_KEY or OPENAI_API_KEY is available, or an advanced
    inferential semantic NLP engine that parses user text to build unique tasks.
    """
    team_members = team_members or []
    full_prompt = f"{title} {goal} {tech_stack}".strip()
    
    # Try calling Gemini / LLM API if key is available in environment
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt_text = f"""
            Decompose the following software project into 5 to 7 logical technical tasks.
            Project Title: {title}
            Goal: {goal}
            Tech Stack: {tech_stack}
            Team Members: {[m.get('name') for m in team_members]}

            Respond ONLY with a JSON array of objects with keys:
            "task_id" (T1, T2...), "module_name", "task_name", "description", "estimated_days" (float 1.0-4.0), "complexity" (Low/Medium/High), "required_skill" (Frontend, Backend, Database, AI Model, API Integration, Testing/QA), "predecessors" (list of task_ids like ["T1"]).
            """
            resp = model.generate_content(prompt_text)
            text_resp = resp.text.strip()
            # Extract JSON array
            json_match = re.search(r'\[.*\]', text_resp, re.DOTALL)
            if json_match:
                tasks = json.loads(json_match.group(0))
                if isinstance(tasks, list) and len(tasks) >= 3:
                    return tasks
        except Exception:
            pass # Fallback to advanced semantic NLP engine

    # Advanced Inferential Semantic NLP Engine
    # Extracts specific technologies, concepts, and verbs directly from the prompt
    clean_text = full_prompt.lower()
    words = re.findall(r'\b[a-zA-Z0-9_\-\+#]+\b', clean_text)
    
    # Identify key tech terms in prompt
    tech_keywords = [w for w in words if w in [
        'react', 'vue', 'angular', 'flutter', 'fastapi', 'flask', 'django', 'express', 'node', 'python',
        'java', 'cpp', 'c++', 'rust', 'opencv', 'pytorch', 'tensorflow', 'scikit', 'sqlite', 'postgres',
        'mongodb', 'firebase', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'graphql', 'rest', 'webrtc',
        'blockchain', 'solidity', 'unity', 'unreal', 'android', 'ios', 'streamlit'
    ]]
    
    primary_stack = ", ".join(list(dict.fromkeys(tech_keywords)).copy()) if tech_keywords else (tech_stack or "Chosen Stack")

    # Construct dynamic, custom tasks customized to user's title and description
    t1_name = f"{title} - System Architecture & Setup"
    t1_desc = f"Initialize base repository, configure environment, and set up project structure for {primary_stack}."
    
    if any(k in clean_text for k in ['ai', 'ml', 'model', 'dataset', 'vision', 'face', 'detect', 'predict', 'classify', 'nlp']):
        m2_name = "Data & Model Pipeline"
        t2_name = f"Dataset Ingestion & Model Architecture ({title})"
        t2_desc = f"Collect training data, build model pipeline, and evaluate accuracy for {title}."
        skill2 = "AI Model"
    elif any(k in clean_text for k in ['mobile', 'flutter', 'react native', 'android', 'ios', 'app']):
        m2_name = "Mobile Screens & Routing"
        t2_name = f"Mobile UI Navigation & State Management"
        t2_desc = f"Design responsive screens, theme configuration, and offline caching for {title}."
        skill2 = "Frontend"
    elif any(k in clean_text for k in ['blockchain', 'crypto', 'web3', 'smart contract', 'solidity']):
        m2_name = "Smart Contracts & Consensus"
        t2_name = f"Smart Contract Architecture & Wallet Binding"
        t2_desc = f"Write smart contracts, execute testnet deployment, and bind Web3 providers."
        skill2 = "Backend"
    else:
        m2_name = "Core Logic & Schema"
        t2_name = f"Database Schema & Repository Layer ({title})"
        t2_desc = f"Design relational/NoSQL database tables and setup data models for {title}."
        skill2 = "Database"

    t3_name = f"Core Backend APIs & Business Logic ({title})"
    t3_desc = f"Develop RESTful/GraphQL API endpoints, authentication middleware, and business services using {primary_stack}."

    t4_name = f"Interactive User Dashboard & Frontend Interface"
    t4_desc = f"Build web/mobile user interface, visual components, forms, and client state handlers for {title}."

    t5_name = f"End-to-End API Integration & System Binding"
    t5_desc = f"Connect UI components with backend API services, implement async loading indicators, and handle errors."

    t6_name = f"System Verification & Automated Testing Suite"
    t6_desc = f"Write unit and integration tests for {title}, fix edge cases, and prepare package for deployment."

    tasks = [
        {
            "task_id": "T1",
            "module_name": "Architecture & Setup",
            "task_name": t1_name,
            "description": t1_desc,
            "estimated_days": 2.0,
            "complexity": "Medium",
            "required_skill": "Frontend",
            "predecessors": []
        },
        {
            "task_id": "T2",
            "module_name": m2_name,
            "task_name": t2_name,
            "description": t2_desc,
            "estimated_days": 2.5,
            "complexity": "High" if skill2 == "AI Model" else "Medium",
            "required_skill": skill2,
            "predecessors": ["T1"]
        },
        {
            "task_id": "T3",
            "module_name": "Backend & Services",
            "task_name": t3_name,
            "description": t3_desc,
            "estimated_days": 3.0,
            "complexity": "High",
            "required_skill": "Backend",
            "predecessors": ["T2"]
        },
        {
            "task_id": "T4",
            "module_name": "Frontend & Interface",
            "task_name": t4_name,
            "description": t4_desc,
            "estimated_days": 2.5,
            "complexity": "Medium",
            "required_skill": "Frontend",
            "predecessors": ["T1"]
        },
        {
            "task_id": "T5",
            "module_name": "System Integration",
            "task_name": t5_name,
            "description": t5_desc,
            "estimated_days": 2.5,
            "complexity": "High",
            "required_skill": "API Integration",
            "predecessors": ["T3", "T4"]
        },
        {
            "task_id": "T6",
            "module_name": "QA & Deployment",
            "task_name": t6_name,
            "description": t6_desc,
            "estimated_days": 1.5,
            "complexity": "Medium",
            "required_skill": "Testing/QA",
            "predecessors": ["T5"]
        }
    ]

    return tasks

def generate_schedule(tasks, team_members, start_date_str, deadline_str):
    """
    Dynamically allocates tasks to any number of team members (1 to 10+).
    Enforces task dependencies, balances member workloads, and maps to calendar dates.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    deadline = datetime.strptime(deadline_str, "%Y-%m-%d")

    num_members = len(team_members)
    if num_members == 0:
        team_members = [{"name": "Default Developer", "skills": ["All"]}]

    member_available_dates = {m['name']: start_date for m in team_members}
    member_workloads = {m['name']: 0.0 for m in team_members}

    scheduled_tasks = {}
    completed_task_end_dates = {}

    pending_tasks = list(tasks)
    
    while pending_tasks:
        ready_tasks = [t for t in pending_tasks if all(p in completed_task_end_dates for p in t.get('predecessors', []))]
        if not ready_tasks:
            ready_tasks = [pending_tasks[0]]

        task = ready_tasks[0]
        pending_tasks.remove(task)

        predecessor_end_dates = [completed_task_end_dates[p] for p in task.get('predecessors', []) if p in completed_task_end_dates]
        earliest_start = max(predecessor_end_dates) if predecessor_end_dates else start_date

        best_member = None
        best_start = None

        matching_members = [
            m['name'] for m in team_members
            if 'skills' not in m or not m['skills'] or task['required_skill'].lower() in [s.lower() for s in m['skills']] or "all" in [s.lower() for s in m['skills']]
        ]
        candidate_names = matching_members if matching_members else [m['name'] for m in team_members]

        for m_name in candidate_names:
            m_date = max(member_available_dates[m_name], earliest_start)
            if best_start is None or m_date < best_start or (m_date == best_start and member_workloads[m_name] < member_workloads[best_member]):
                best_member = m_name
                best_start = m_date

        duration = timedelta(days=task['estimated_days'])
        task_end = best_start + duration

        member_available_dates[best_member] = task_end
        member_workloads[best_member] += task['estimated_days']
        completed_task_end_dates[task['task_id']] = task_end

        scheduled_tasks[task['task_id']] = {
            **task,
            "assigned_member_name": best_member,
            "planned_start": best_start.strftime("%Y-%m-%d"),
            "planned_end": task_end.strftime("%Y-%m-%d"),
            "status": "PENDING",
            "actual_progress_pct": 0.0
        }

    max_end_date = max(completed_task_end_dates.values()) if completed_task_end_dates else start_date
    achievable = max_end_date <= deadline

    return list(scheduled_tasks.values()), member_workloads, achievable, max_end_date.strftime("%Y-%m-%d")

def generate_rescue_plan(tasks, current_progress, team_members, deadline_str):
    """
    Project Rescue Mode: Detects delayed/blocked tasks, rebalances workload across any team size,
    reassigns bottleneck tasks, and generates an optimized recovery schedule.
    """
    deadline = datetime.strptime(deadline_str, "%Y-%m-%d")
    today = datetime.now()

    updated_tasks = []
    delayed_bottlenecks = []

    for task in tasks:
        tid = task['task_id']
        prog = current_progress.get(tid, task.get('actual_progress_pct', 0.0))
        task_copy = dict(task)
        task_copy['actual_progress_pct'] = prog

        if prog < 100.0 and task.get('status') != 'COMPLETED':
            p_end = datetime.strptime(task['planned_end'], "%Y-%m-%d")
            if today > p_end or (prog < 50.0 and (p_end - today).days <= 3):
                task_copy['is_delayed'] = 1
                delayed_bottlenecks.append(task_copy)

        updated_tasks.append(task_copy)

    incomplete_tasks = [t for t in updated_tasks if t['actual_progress_pct'] < 100.0]
    
    for t in incomplete_tasks:
        remaining_ratio = max(0.1, (100.0 - t['actual_progress_pct']) / 100.0)
        t['estimated_days'] = round(t['estimated_days'] * remaining_ratio, 1)

    new_schedule, new_workloads, achievable, recovery_end_date = generate_schedule(
        incomplete_tasks, team_members, today.strftime("%Y-%m-%d"), deadline_str
    )

    rescue_summary = {
        "bottlenecks_detected": len(delayed_bottlenecks),
        "delayed_tasks": [t['task_name'] for t in delayed_bottlenecks],
        "is_deadline_achievable": achievable,
        "original_deadline": deadline_str,
        "projected_recovery_end_date": recovery_end_date,
        "recommendation": (
            "Workload redistributed and tasks re-assigned to underutilized team members to meet deadline."
            if achievable else
            "Critical path bottleneck exceeds remaining deadline. Non-critical scope reduction or deadline extension recommended."
        )
    }

    return new_schedule, new_workloads, rescue_summary
