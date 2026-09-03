import os
import json
import sqlite3
from datetime import datetime

def get_db():
    """Returns SQLite connection with WAL mode enabled and extended timeout for concurrent access."""
    target_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(target_dir, exist_ok=True)
    db_path = os.path.join(target_dir, "projectpilot.db")
    conn = sqlite3.connect(db_path, timeout=60.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    """Initializes schema tables for persistent project and agent state."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Projects table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS PROJECTS (
            project_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            goal TEXT NOT NULL,
            start_date TEXT NOT NULL,
            deadline TEXT NOT NULL,
            tech_stack TEXT,
            status TEXT DEFAULT 'ACTIVE',
            active_plan_version INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Dynamic Team Members table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS TEAM_MEMBERS (
            member_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            skills TEXT,
            FOREIGN KEY (project_id) REFERENCES PROJECTS(project_id) ON DELETE CASCADE
        )
        """)

        # Tasks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS TASKS (
            task_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            module_name TEXT NOT NULL,
            task_name TEXT NOT NULL,
            description TEXT,
            assigned_member_name TEXT,
            estimated_days REAL NOT NULL,
            complexity TEXT DEFAULT 'Medium',
            required_skill TEXT,
            planned_start TEXT,
            planned_end TEXT,
            status TEXT DEFAULT 'PENDING',
            actual_progress_pct REAL DEFAULT 0.0,
            plan_version INTEGER DEFAULT 1,
            FOREIGN KEY (project_id) REFERENCES PROJECTS(project_id) ON DELETE CASCADE
        )
        """)

        # Task Dependencies table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS DEPENDENCIES (
            dep_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            predecessor_task_id TEXT NOT NULL,
            plan_version INTEGER DEFAULT 1,
            FOREIGN KEY (project_id) REFERENCES PROJECTS(project_id) ON DELETE CASCADE
        )
        """)

        # Agent Decisions & Activity Feed
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS AGENT_DECISIONS (
            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            action TEXT NOT NULL,
            reasoning TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Auto-migrate metadata column if missing from earlier table creation
        try:
            cursor.execute("ALTER TABLE AGENT_DECISIONS ADD COLUMN metadata TEXT")
        except sqlite3.OperationalError:
            pass

        # Project Rescue & Human Approvals
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS APPROVALS (
            approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            plan_version INTEGER NOT NULL,
            action_requested TEXT NOT NULL,
            status TEXT DEFAULT 'PENDING',
            user_comment TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Technical Research Resources (ChromaDB RAG)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS RESEARCH_RESOURCES (
            resource_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            query TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Test Results Log Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS TEST_RESULTS (
            test_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            test_name TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()

def log_agent_decision(project_id, agent_name, action, reasoning, metadata=None, conn=None):
    """Logs an agentic action into AGENT_DECISIONS, reusing active connection if passed."""
    meta_json = json.dumps(metadata) if metadata else None
    if conn is not None:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO AGENT_DECISIONS (project_id, agent_name, action, reasoning, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (project_id, agent_name, action, reasoning, meta_json))
        conn.commit()
    else:
        with get_db() as c:
            cursor = c.cursor()
            cursor.execute("""
                INSERT INTO AGENT_DECISIONS (project_id, agent_name, action, reasoning, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, agent_name, action, reasoning, meta_json))
            c.commit()

def get_agent_logs(project_id):
    """Returns agent decision log history for a project."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM AGENT_DECISIONS WHERE project_id = ? ORDER BY decision_id DESC", (project_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
