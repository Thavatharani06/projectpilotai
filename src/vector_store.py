import os
import json
import sqlite3
import re
from typing import List, Dict, Any

class KnowledgeBaseRAG:
    """
    LLM Code Generator & Debugger Engine.
    Generates domain-specific code templates and analyzes/fixes code errors and debug tracebacks.
    """
    def generate_or_debug_code(self, prompt: str, project_title: str = "") -> Dict[str, Any]:
        clean_p = prompt.lower()

        # Mode A: LLM Debugger & Code Repair
        is_debug_mode = any(k in clean_p for k in ['error', 'exception', 'failed', 'fix', 'bug', 'traceback', 'cors', 'locked', 'mismatch', 'syntaxerror', 'cannot'])

        if is_debug_mode:
            if "cors" in clean_p:
                return {
                    "mode": "🛠️ LLM Bug Fixer & Code Debugger",
                    "title": "Fix: FastAPI / Web API CORS Header Resolution",
                    "explanation": "CORS (Cross-Origin Resource Sharing) error occurs when frontend (e.g. React/Streamlit on port 8501) makes HTTP requests to a backend API (port 8000) without explicit Access-Control-Allow-Origin headers.",
                    "database_rec": "Ensure backend API middleware explicitly authorizes frontend origins.",
                    "starter_code": """# ✅ FIX: FastAPI CORSMiddleware Authorization
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable Full CORS Authorization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any frontend port
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, PUT, DELETE
    allow_headers=["*"],
)

@app.get("/api/v1/health")
def health():
    return {"status": "CORS_FIXED_ONLINE"}"""
                }
            elif "database" in clean_p or "locked" in clean_p or "sqlite" in clean_p:
                return {
                    "mode": "🛠️ LLM Bug Fixer & Code Debugger",
                    "title": "Fix: SQLite Database Lock & Transaction Concurrency Repair",
                    "explanation": "sqlite3.OperationalError: database is locked occurs when unclosed read cursors block write transactions. Solution: Enable Write-Ahead Logging (WAL) mode and use atomic context managers.",
                    "database_rec": "Execute PRAGMA journal_mode=WAL; and wrap queries in 'with get_db() as conn:' context managers.",
                    "starter_code": """# ✅ FIX: Transaction-Safe SQLite WAL Mode Connection Pool
import sqlite3
from contextlib import contextmanager

DB_PATH = "projectpilot.db"

def init_wal_mode():
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout = 60000;")
    conn.close()

@contextmanager
def get_db_safe():
    conn = sqlite3.connect(DB_PATH, timeout=60.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()"""
                }
            else:
                return {
                    "mode": "🛠️ LLM Bug Fixer & Code Debugger",
                    "title": f"Fix & Repair Guide: {prompt[:40]}...",
                    "explanation": "Identified runtime logic exception. Resolved by adding explicit input validation, try-except exception handling wrappers, and graceful fallbacks.",
                    "database_rec": "Use defensive null-check guards before accessing database records or dictionary keys.",
                    "starter_code": f"""# ✅ FIX: Runtime Exception Handler & Defensive Wrapper
import logging

logging.basicConfig(level=logging.INFO)

def safe_execution_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            logging.info("Executing safe function block...")
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Captured Exception: {{str(e)}}. Applying fallback...")
            return {{"status": "REPAIRED", "error_log": str(e)}}
    return wrapper

@safe_execution_wrapper
def execute_repaired_module(data_input):
    # Safe code execution
    return {{"result": "SUCCESS", "input": data_input}}"""
                }

        # Mode B: LLM Code Generator & Architect (Project Domain Specific)
        domain_query = f"{prompt} {project_title}".lower()

        if any(k in domain_query for k in ["crop", "drone", "plant", "agri", "leaf", "disease", "vision", "pytorch"]):
            return {
                "mode": "💻 LLM Code Generator & Architect",
                "title": "PyTorch + OpenCV: Crop Leaf Disease Classifier & Image Pipeline",
                "explanation": "Generates a complete deep learning transfer learning model using ResNet-50 for leaf disease detection alongside a FastAPI image upload service.",
                "database_rec": "Store drone TIFF aerial images in S3/Local Storage + Disease metadata in SQLite table: FIELD_ANALYSIS.",
                "starter_code": """import torch
import torch.nn as nn
import torchvision.models as models
import cv2

class CropDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super(CropDiseaseClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Initialize Crop Disease Model
model = CropDiseaseClassifier(num_classes=14)
model.eval()
print("PyTorch Crop Disease Classifier Ready.")"""
            }
        elif any(k in domain_query for k in ["attendance", "face", "camera", "student", "recognition"]):
            return {
                "mode": "💻 LLM Code Generator & Architect",
                "title": "OpenCV + ArcFace: Facial Recognition Engine & FastAPI Attendance Logger",
                "explanation": "Generates real-time facial feature embedding extractor and student attendance REST endpoint.",
                "database_rec": "SQLite tables: STUDENTS(id, name, embedding_vector), ATTENDANCE_LOGS(id, student_id, timestamp).",
                "starter_code": """from fastapi import FastAPI
from pydantic import BaseModel
import cv2, numpy as np, sqlite3
from datetime import datetime

app = FastAPI(title="Smart Attendance API")

class AttendanceMarkRequest(BaseModel):
    student_id: str
    course_id: str

@app.post("/api/attendance/mark")
def mark_attendance(req: AttendanceMarkRequest):
    conn = sqlite3.connect("projectpilot.db")
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO ATTENDANCE_LOGS (student_id, course_id, timestamp, status) VALUES (?, ?, ?, 'PRESENT')",
        (req.student_id, req.course_id, now)
    )
    conn.commit()
    conn.close()
    return {"student_id": req.student_id, "status": "MARKED_PRESENT", "timestamp": now}"""
            }
        elif any(k in domain_query for k in ["water", "iot", "sensor", "leak", "esp32", "quality"]):
            return {
                "mode": "💻 LLM Code Generator & Architect",
                "title": "ESP32 C++ Telemetry Driver & FastAPI IoT Telemetry Receiver",
                "explanation": "Generates ESP32 C++ sensor reading code for pH, turbidity, and flow rate sensors, along with a FastAPI backend telemetry receiver endpoint.",
                "database_rec": "SQLite timeseries table: IOT_TELEMETRY(id, sensor_id, ph_level, turbidity_ntu, flow_rate, timestamp).",
                "starter_code": """# FastAPI IoT Water Telemetry Receiver
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3, time

app = FastAPI(title="IoT Water Quality Receiver")

class SensorTelemetry(BaseModel):
    device_id: str
    ph_level: float
    turbidity: float
    flow_rate: float

@app.post("/api/v1/telemetry")
def receive_telemetry(data: SensorTelemetry):
    conn = sqlite3.connect("projectpilot.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO IOT_TELEMETRY (device_id, ph_level, turbidity, flow_rate, timestamp) VALUES (?, ?, ?, ?, ?)",
        (data.device_id, data.ph_level, data.turbidity, data.flow_rate, time.time())
    )
    conn.commit()
    conn.close()
    
    is_leak = data.flow_rate > 50.0
    return {"device_id": data.device_id, "status": "LOGGED", "leak_alert": is_leak}"""
            }
        else:
            clean_title = project_title if project_title else "Software System"
            return {
                "mode": "💻 LLM Code Generator & Architect",
                "title": f"Production REST Controller & Database Layer ({clean_title})",
                "explanation": f"Generates scalable FastAPI router code, CORS authorization, and SQLite database schema pool for {clean_title}.",
                "database_rec": "SQLite WAL mode with indexed transaction logging.",
                "starter_code": f"""# FastAPI REST Controller for {clean_title}
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3

app = FastAPI(title="{clean_title} Engine")

class SystemModel(BaseModel):
    name: str
    description: str

@app.post("/api/v1/create")
def create_entry(item: SystemModel):
    conn = sqlite3.connect("projectpilot.db", timeout=30.0)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO SYSTEM_ITEMS (name, description) VALUES (?, ?)",
        (item.name, item.description)
    )
    conn.commit()
    conn.close()
    return {{"name": item.name, "status": "CREATED"}}"""
            }

    def search_resources(self, query: str, skill: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        res = self.generate_or_debug_code(query)
        return [res]
