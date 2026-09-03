import os
import json
import re
from datetime import datetime
from src.database import get_db, log_agent_decision
from src.scheduler import decompose_project_goal, generate_schedule, generate_rescue_plan
from src.vector_store import KnowledgeBaseRAG

class ProjectPilotOrchestrator:
    def __init__(self):
        self.rag = KnowledgeBaseRAG()

    def run_scispace_research_copilot(self, project_id, user_query):
        """Fetches project-tailored literature papers, public repos, arXiv preprints, database recommendations, and starter code safely."""
        resources = self.rag.search_resources(user_query, top_k=3)
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM RESEARCH_RESOURCES WHERE project_id = ?", (project_id,))

            for res in resources:
                title = res.get('title', 'Technical Specification')
                url = res.get('url', f"https://github.com/search?q={user_query.replace(' ', '+')}")
                cat = res.get('category', 'Technical Resource')
                summary = res.get('summary', res.get('explanation', 'Technical guideline and system architecture spec.'))
                db_rec = res.get('database_rec', 'SQLite for MVP, PostgreSQL for cloud.')

                cursor.execute("""
                    INSERT INTO RESEARCH_RESOURCES (project_id, query, title, url, summary)
                    VALUES (?, ?, ?, ?, ?)
                """, (project_id, user_query, title, url, f"[{cat}] {summary} | DB Spec: {db_rec}"))

            conn.commit()

        log_agent_decision(
            project_id=project_id,
            agent_name="CHITTI (Research Agent)",
            action="Retrieved Relevant Papers & Public GitHub Repositories",
            reasoning=f"Retrieved {len(resources)} project-tailored papers, GitHub open-source repos, arXiv preprints, and database specs for '{user_query}'."
        )

        return resources

    def run_initial_planning(self, project_id, title, goal, start_date, deadline, tech_stack, team_members):
        """Planner Agent: Decomposes goal, allocates work dynamically, and populates relevant papers & resources."""
        # 1. Fetch project-tailored IEEE papers & GitHub open-source project repos safely
        self.run_scispace_research_copilot(project_id, f"{title} {goal}")

        # 2. Decompose into modules and tasks
        tasks = decompose_project_goal(title, goal, tech_stack, team_members)
        
        # 3. Schedule dynamically
        scheduled_tasks, workloads, achievable, end_date = generate_schedule(tasks, team_members, start_date, deadline)

        # 4. Store project, team members, tasks & dependencies in database
        with get_db() as conn:
            cursor = conn.cursor()

            # Save project
            cursor.execute("""
                INSERT OR REPLACE INTO PROJECTS (project_id, title, goal, start_date, deadline, tech_stack, status, active_plan_version)
                VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', 1)
            """, (project_id, title, goal, start_date, deadline, tech_stack))

            # Save team members with project-scoped unique IDs
            cursor.execute("DELETE FROM TEAM_MEMBERS WHERE project_id = ?", (project_id,))
            for idx, m in enumerate(team_members):
                m_id = f"{project_id}_MEMBER_{idx+1}"
                cursor.execute("""
                    INSERT OR REPLACE INTO TEAM_MEMBERS (member_id, project_id, name, skills)
                    VALUES (?, ?, ?, ?)
                """, (m_id, project_id, m["name"], json.dumps(m.get("skills", []))))

            # Store tasks & dependencies
            cursor.execute("DELETE FROM TASKS WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM DEPENDENCIES WHERE project_id = ?", (project_id,))

            for t in scheduled_tasks:
                full_task_id = f"{project_id}_{t['task_id']}"
                cursor.execute("""
                    INSERT OR REPLACE INTO TASKS (task_id, project_id, module_name, task_name, description, assigned_member_name, estimated_days, complexity, required_skill, planned_start, planned_end, status, actual_progress_pct, plan_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (full_task_id, project_id, t['module_name'], t['task_name'], t['description'], t['assigned_member_name'], t['estimated_days'], t['complexity'], t['required_skill'], t['planned_start'], t['planned_end'], t['status'], t['actual_progress_pct']))

                for pred_id in t.get('predecessors', []):
                    cursor.execute("""
                        INSERT INTO DEPENDENCIES (project_id, task_id, predecessor_task_id, plan_version)
                        VALUES (?, ?, ?, 1)
                    """, (project_id, full_task_id, f"{project_id}_{pred_id}"))

            conn.commit()

        # Log Planner Agent Action
        log_agent_decision(
            project_id=project_id,
            agent_name="CHITTI (Planner Agent)",
            action="Created Initial Project Roadmap",
            reasoning=f"Decomposed goal into {len(scheduled_tasks)} modules/tasks. Balanced workload across {len(team_members)} members. Projected completion: {end_date} (Achievable: {achievable}).",
            metadata={"workloads": workloads, "projected_end": end_date}
        )

        return scheduled_tasks, workloads, achievable, end_date

    def process_natural_language_assistant(self, project_id, prompt_text):
        """CHITTI Assistant: Processes natural language user prompt for technical help or project delay re-planning."""
        clean_p = prompt_text.lower()

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM TASKS WHERE project_id = ?", (project_id,))
            tasks = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT * FROM TEAM_MEMBERS WHERE project_id = ?", (project_id,))
            members = [dict(r) for r in cursor.fetchall()]

        # Check if prompt mentions delay, illness, or bottleneck for a member or task
        is_delay_issue = any(k in clean_p for k in ['delayed', 'stuck', 'sick', 'error', 'failing', 'behind', 'bottleneck', 'cannot finish', 'reassign'])
        
        if is_delay_issue and tasks:
            target_task = None
            for t in tasks:
                t_name_clean = t['task_name'].lower()
                m_name_clean = t['assigned_member_name'].lower()
                if m_name_clean in clean_p or any(w in clean_p for w in t_name_clean.split()):
                    target_task = t
                    break
            
            if not target_task:
                target_task = tasks[1] if len(tasks) > 1 else tasks[0]

            sim_updates = {target_task['task_id']: 30.0}
            eval_res = self.run_reviewer_agent(project_id, sim_updates)
            
            log_agent_decision(
                project_id=project_id,
                agent_name="CHITTI (Rescue Assistant)",
                action="Processed Natural Language Blocker",
                reasoning=f"Student reported: '{prompt_text}'. Flagged bottleneck on task '{target_task['task_name']}' assigned to {target_task['assigned_member_name']}. Triggered Project Rescue Mode."
            )
            
            return {
                "type": "RESCUE_TRIGGERED",
                "message": f"Reviewer Agent detected bottleneck on '{target_task['task_name']}' ({target_task['assigned_member_name']}). CHITTI generated a recovery plan.",
                "evaluation": eval_res
            }
        else:
            rag_res = self.rag.generate_or_debug_code(prompt_text)
            
            log_agent_decision(
                project_id=project_id,
                agent_name="CHITTI (Technical Assistant)",
                action="Answered Student Technical Query",
                reasoning=f"Answered question: '{prompt_text}' using CHITTI LLM Code Generator & Debugger engine."
            )
            
            return {
                "type": "TECHNICAL_GUIDANCE",
                "message": f"CHITTI Guidance for '{prompt_text}':\n\n{rag_res.get('explanation', '')}",
                "resources": [rag_res]
            }

    def run_reviewer_agent(self, project_id, progress_updates=None):
        """Reviewer Agent: Monitors progress, computes health score, detects delays/risks, and triggers rescue mode if needed."""
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM PROJECTS WHERE project_id = ?", (project_id,))
            proj_r = cursor.fetchone()
            if not proj_r:
                return {"risk_level": "LOW", "rescue_mode_triggered": False}
            proj = dict(proj_r)

            cursor.execute("SELECT * FROM TASKS WHERE project_id = ?", (project_id,))
            tasks = [dict(r) for r in cursor.fetchall()]

            cursor.execute("SELECT * FROM TEAM_MEMBERS WHERE project_id = ?", (project_id,))
            members = [dict(r) for r in cursor.fetchall()]

            if progress_updates:
                for t in tasks:
                    t_raw_id = t['task_id'].replace(f"{project_id}_", "")
                    if t['task_id'] in progress_updates or t_raw_id in progress_updates:
                        pct = progress_updates.get(t['task_id'], progress_updates.get(t_raw_id))
                        t['actual_progress_pct'] = pct
                        status = "COMPLETED" if pct >= 100.0 else ("IN_PROGRESS" if pct > 0 else "PENDING")
                        cursor.execute("UPDATE TASKS SET actual_progress_pct = ?, status = ? WHERE project_id = ? AND task_id = ?", (pct, status, project_id, t['task_id']))
                conn.commit()

        total_tasks = len(tasks)
        completed_tasks = sum(1 for t in tasks if t['actual_progress_pct'] >= 100.0)
        overall_progress = round(sum(t['actual_progress_pct'] for t in tasks) / total_tasks, 1) if total_tasks > 0 else 0.0

        today = datetime.now()
        delayed_tasks = []
        for t in tasks:
            p_end = datetime.strptime(t['planned_end'], "%Y-%m-%d")
            if t['actual_progress_pct'] < 100.0 and (today > p_end or (t['actual_progress_pct'] < 40.0 and (p_end - today).days <= 2)):
                delayed_tasks.append(t)

        risk_level = "LOW"
        if len(delayed_tasks) >= 2 or (len(delayed_tasks) >= 1 and overall_progress < 40.0):
            risk_level = "HIGH"
        elif len(delayed_tasks) == 1:
            risk_level = "MEDIUM"

        log_agent_decision(
            project_id=project_id,
            agent_name="CHITTI (Reviewer Agent)",
            action="Evaluated Project Health",
            reasoning=f"Overall Progress: {overall_progress}%. Completed: {completed_tasks}/{total_tasks} tasks. Delayed Tasks Detected: {len(delayed_tasks)}. Assessed Risk Level: {risk_level}."
        )

        rescue_plan = None
        if risk_level == "HIGH":
            rescue_plan = self.trigger_rescue_mode(project_id, proj, tasks, members)

        return {
            "overall_progress": overall_progress,
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "delayed_tasks_count": len(delayed_tasks),
            "delayed_task_names": [t['task_name'] for t in delayed_tasks],
            "risk_level": risk_level,
            "rescue_mode_triggered": risk_level == "HIGH",
            "rescue_plan": rescue_plan
        }

    def trigger_rescue_mode(self, project_id, proj, tasks, members):
        """Triggers Rescue Mode: CHITTI Planner Agent re-analyzes bottlenecks and generates recovery schedule for human approval."""
        current_progress = {t['task_id']: t['actual_progress_pct'] for t in tasks}
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT task_id, predecessor_task_id FROM DEPENDENCIES WHERE project_id = ?", (project_id,))
            dep_rows = cursor.fetchall()
            
            preds_by_task = {}
            for r in dep_rows:
                preds_by_task.setdefault(r['task_id'], []).append(r['predecessor_task_id'])

            task_list_for_sched = []
            for t in tasks:
                t_copy = dict(t)
                t_copy['predecessors'] = preds_by_task.get(t['task_id'], [])
                task_list_for_sched.append(t_copy)

            new_schedule, new_workloads, rescue_summary = generate_rescue_plan(
                task_list_for_sched, current_progress, members, proj['deadline']
            )

            cursor.execute("""
                INSERT INTO APPROVALS (project_id, plan_version, action_requested, status, user_comment)
                VALUES (?, ?, ?, 'PENDING', ?)
            """, (project_id, proj['active_plan_version'] + 1, "PROJECT_RESCUE_REPLAN", rescue_summary['recommendation']))

            conn.commit()

        log_agent_decision(
            project_id=project_id,
            agent_name="CHITTI (Planner Agent)",
            action="Generated Project Rescue Plan",
            reasoning=f"Detected bottleneck on delayed tasks. Re-allocated workload across {len(members)} team members. Projected new completion date: {rescue_summary['projected_recovery_end_date']}. Pending Human Approval."
        )

        return {
            "summary": rescue_summary,
            "proposed_schedule": new_schedule,
            "proposed_workloads": new_workloads
        }

    def handle_rescue_approval(self, project_id, action):
        """Applies human decision (APPROVE, REJECT, EDIT) to the proposed rescue plan."""
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT active_plan_version FROM PROJECTS WHERE project_id = ?", (project_id,))
            row = cursor.fetchone()
            curr_ver = row['active_plan_version'] if row else 1

            if action == "APPROVE":
                new_ver = curr_ver + 1
                cursor.execute("UPDATE PROJECTS SET active_plan_version = ? WHERE project_id = ?", (new_ver, project_id))
                cursor.execute("UPDATE APPROVALS SET status = 'APPROVED' WHERE project_id = ? AND status = 'PENDING'", (project_id,))
                
                log_agent_decision(
                    project_id=project_id,
                    agent_name="CHITTI (Orchestrator)",
                    action="Rescue Plan Approved & Activated",
                    reasoning=f"User APPROVED the recovery schedule. Project active plan updated to Version {new_ver}."
                )
            elif action == "REJECT":
                cursor.execute("UPDATE APPROVALS SET status = 'REJECTED' WHERE project_id = ? AND status = 'PENDING'", (project_id,))
                log_agent_decision(
                    project_id=project_id,
                    agent_name="CHITTI (Orchestrator)",
                    action="Rescue Plan Rejected",
                    reasoning="User REJECTED the recovery schedule. Retaining existing schedule."
                )

            conn.commit()
