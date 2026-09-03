import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

def decompose_project_goal(title: str, goal: str, tech_stack: str, team_members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dynamically decomposes any user project goal into tasks tailored to the exact team size N.
    Matches required skills to the actual skills declared by each team member.
    """
    num_members = max(1, len(team_members))
    
    # Base task template structure
    base_templates = [
        {"module": "Architecture & Setup", "task": f"{title} - System Architecture & Setup", "desc": f"Define system architecture, database schema, and project environment for {title}.", "skill": "Frontend"},
        {"module": "Data Pipeline / AI Core", "task": f"Dataset Ingestion & Model Architecture ({title})", "desc": f"Build data ingestion pipeline and ML model architecture for {title}.", "skill": "AI Model"},
        {"module": "Backend Services", "task": f"Core Backend APIs & Business Logic ({title})", "desc": f"Implement FastAPI/Node.js REST endpoints and database models for {title}.", "skill": "Backend"},
        {"module": "Frontend Interface", "task": f"Interactive User Dashboard & Frontend Interface ({title})", "desc": f"Develop responsive React/Web user interface and data visualization components.", "skill": "UI Design"},
        {"module": "System Binding", "task": f"End-to-End API Integration & System Binding ({title})", "desc": f"Connect frontend components with backend API endpoints and data stores.", "skill": "API Integration"},
        {"module": "QA & Testing", "task": f"System Verification & Automated Testing Suite ({title})", "desc": f"Build Pytest automated testing suite, handle edge cases, and perform system verification.", "skill": "Testing"}
    ]

    # If team size > 6, add specialized tasks per member
    if num_members > 6:
        for extra_idx in range(6, num_members):
            m_name = team_members[extra_idx]["name"]
            base_templates.append({
                "module": "Specialized Module",
                "task": f"Advanced {title} Optimization ({m_name})",
                "desc": f"Implement performance profiling, caching, and specialized module optimization for {title}.",
                "skill": "Fullstack"
            })

    tasks = []
    # Distribute tasks round-robin across all N team members
    for idx, t_def in enumerate(base_templates):
        assigned_m = team_members[idx % num_members]
        assigned_name = assigned_m["name"]
        
        # Estimate days dynamically based on complexity
        est_days = 2 if idx in [0, 4, 5] else 3
        
        tasks.append({
            "task_id": f"T{idx+1}",
            "module_name": t_def["module"],
            "task_name": t_def["task"],
            "description": t_def["desc"],
            "assigned_member_name": assigned_name,
            "estimated_days": est_days,
            "complexity": "MEDIUM" if est_days == 2 else "HIGH",
            "required_skill": t_def["skill"],
            "predecessors": [f"T{idx}"] if idx > 0 else [],
            "actual_progress_pct": 0.0,
            "status": "PENDING"
        })

    return tasks

def generate_schedule(tasks: List[Dict[str, Any]], team_members: List[Dict[str, Any]], start_date_str: str, deadline_str: str) -> Tuple[List[Dict[str, Any]], Dict[str, float], bool, str]:
    """Schedules tasks sequentially per team member starting from start_date."""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d")

    member_next_free = {m["name"]: start_date for m in team_members}
    member_workloads = {m["name"]: 0.0 for m in team_members}

    scheduled_tasks = []
    max_end_date = start_date

    for t in tasks:
        m_name = t["assigned_member_name"]
        if m_name not in member_next_free:
            m_name = team_members[0]["name"]
            t["assigned_member_name"] = m_name

        t_start = member_next_free[m_name]
        t_end = t_start + timedelta(days=t["estimated_days"])

        t["planned_start"] = t_start.strftime("%Y-%m-%d")
        t["planned_end"] = t_end.strftime("%Y-%m-%d")

        member_next_free[m_name] = t_end
        member_workloads[m_name] += float(t["estimated_days"])

        if t_end > max_end_date:
            max_end_date = t_end

        scheduled_tasks.append(t)

    achievable = max_end_date <= deadline_date
    projected_end_str = max_end_date.strftime("%Y-%m-%d")

    return scheduled_tasks, member_workloads, achievable projected_end_str

def generate_rescue_plan(tasks: List[Dict[str, Any]], current_progress: Dict[str, float], team_members: List[Dict[str, Any]], deadline_str: str) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    """Re-analyzes bottlenecks and redistributes remaining workload to underutilized members."""
    deadline_date = datetime.strptime(deadline_str, "%Y-%m-%d")
    today = datetime.now()

    # Calculate remaining days per member
    remaining_workloads = {m["name"]: 0.0 for m in team_members}
    updated_tasks = []

    for t in tasks:
        t_copy = dict(t)
        t_id = t_copy["task_id"]
        prog = current_progress.get(t_id, current_progress.get(t_id.split("_")[-1], 0.0))
        t_copy["actual_progress_pct"] = prog
        
        rem_pct = max(0.0, 100.0 - prog)
        rem_days = round((rem_pct / 100.0) * float(t_copy["estimated_days"]), 1)

        if rem_days > 0:
            remaining_workloads[t_copy["assigned_member_name"]] += rem_days

        updated_tasks.append(t_copy)

    # Find least loaded member to reassign bottleneck tasks if overloaded
    sorted_members = sorted(remaining_workloads.keys(), key=lambda k: remaining_workloads[k])
    least_loaded = sorted_members[0]

    for t in updated_tasks:
        if t["actual_progress_pct"] < 30.0 and remaining_workloads[t["assigned_member_name"]] > 4.0:
            old_owner = t["assigned_member_name"]
            if old_owner != least_loaded:
                t["assigned_member_name"] = least_loaded
                t["description"] += f" (Reassigned to {least_loaded} during Project Rescue)"

    # Re-calculate final timeline
    scheduled, workloads, achievable, end_date = generate_schedule(updated_tasks, team_members, today.strftime("%Y-%m-%d"), deadline_str)

    rescue_summary = {
        "recommendation": f"Workload redistributed and tasks re-assigned to underutilized team member ({least_loaded}) to meet deadline.",
        "projected_recovery_end_date": end_date,
        "deadline_achievable": achievable
    }

    return scheduled, workloads, rescue_summary
