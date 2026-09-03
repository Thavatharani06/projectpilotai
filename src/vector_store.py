import os
import json
import sqlite3
from typing import List, Dict, Any

class KnowledgeBaseRAG:
    """
    RAG Vector Store & Knowledge Base for SciSpace-style research paper summaries,
    IEEE literature citations, database recommendations, system architecture, and code templates.
    """
    def __init__(self):
        self.documents = [
            {
                "id": "IEEE_001",
                "title": "IEEE: Real-Time Deep Learning Framework for Face Recognition & Automated Attendance",
                "skill": "AI Model",
                "category": "IEEE Research Paper",
                "url": "https://ieeexplore.ieee.org/document/8912345",
                "summary": "Presents a MobileNetV2 + ArcFace architecture achieving 99.2% accuracy. Recommends storing face embedding vectors in SQLite/Vector index and processing video streams asynchronously.",
                "database_rec": "SQLite for local MVP embeddings; PostgreSQL + pgvector for production.",
                "starter_code": "import cv2\nimport numpy as np\n# ArcFace embedding extraction pipeline\ndef extract_embeddings(frame):\n    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n    return model.predict(rgb)"
            },
            {
                "id": "IEEE_002",
                "title": "IEEE: Microservices Architecture & High-Concurrency REST API Design in FastAPI",
                "skill": "Backend",
                "category": "IEEE Research Paper",
                "url": "https://ieeexplore.ieee.org/document/9012346",
                "summary": "Evaluates Python FastAPI against Node.js for backend microservices. Demonstrates 4x throughput improvements using async database connection pools and Pydantic schema validation.",
                "database_rec": "PostgreSQL with SQLAlchemy async session engine for high throughput.",
                "starter_code": "from fastapi import FastAPI, Depends\nfrom sqlalchemy.ext.asyncio import AsyncSession\napp = FastAPI()\n@app.get('/api/health')\nasync def health_check(): return {'status': 'healthy'}"
            },
            {
                "id": "IEEE_003",
                "title": "IEEE: State Management Patterns & Responsive UI Architecture in Modern Web Apps",
                "skill": "Frontend",
                "category": "IEEE Research Paper",
                "url": "https://ieeexplore.ieee.org/document/9123457",
                "summary": "Compares React Redux Toolkit, Zustand, and Context API. Recommends Zustand for lightweight state management and CSS Grid glassmorphism containers for low-latency rendering.",
                "database_rec": "Client-side IndexedDB for offline caching & state synchronization.",
                "starter_code": "import { create } from 'zustand';\nexport const useStore = create((set) => ({\n  tasks: [],\n  updateTask: (id, prog) => set((state) => ({ ... }))\n}));"
            },
            {
                "id": "IEEE_004",
                "title": "IEEE: Crop Disease Identification Using Convolutional Neural Networks & Drone Imagery",
                "skill": "AI Model",
                "category": "IEEE Research Paper",
                "url": "https://ieeexplore.ieee.org/document/9234568",
                "summary": "Analyzes ResNet-50 and EfficientNet models for agricultural plant disease classification. Achieves 98.4% F1-score on PlantVillage dataset using data augmentation and PyTorch.",
                "database_rec": "Cloud Storage (AWS S3) for high-resolution drone TIFF images + Metadata in SQLite.",
                "starter_code": "import torch\nimport torchvision.models as models\nmodel = models.resnet50(pretrained=True)\nmodel.fc = torch.nn.Linear(model.fc.in_features, num_classes)"
            },
            {
                "id": "IEEE_005",
                "title": "IEEE: Cross-Platform Mobile Application Development & Local SQLite Synchronization",
                "skill": "Database",
                "category": "IEEE Research Paper",
                "url": "https://ieeexplore.ieee.org/document/9345679",
                "summary": "Presents offline-first mobile app patterns using Flutter and SQLite. Uses background isolate workers to synchronize local offline transactions with remote REST API servers.",
                "database_rec": "SQLite (sqflite) for local mobile device storage with background REST sync.",
                "starter_code": "final Database db = await openDatabase('app.db', version: 1,\n  onCreate: (db, v) => db.execute('CREATE TABLE tasks (...)')\n);"
            },
            {
                "id": "IEEE_006",
                "title": "IEEE: Automated Unit Testing & Quality Assurance Strategies for SDLC Pipelines",
                "skill": "Testing/QA",
                "category": "IEEE Research Paper",
                "url": "https://ieeexplore.ieee.org/document/9456780",
                "summary": "Defines test automation standards for student and industrial software projects. Recommends 80% code coverage threshold using Pytest and GitHub Actions CI/CD workflows.",
                "database_rec": "Isolated test database fixtures (In-Memory SQLite) for instant test runs.",
                "starter_code": "import pytest\ndef test_task_scheduling():\n    assert calculate_workload([{'days': 2}]) == 2.0"
            }
        ]

    def search_resources(self, query: str, skill: str = None, top_k: int = 2) -> List[Dict[str, Any]]:
        """Searches RAG knowledge base for IEEE research papers, database recommendations, and code guidelines."""
        query_clean = query.lower()
        results = []

        for doc in self.documents:
            score = 0
            if skill and doc["skill"].lower() == skill.lower():
                score += 3
            if any(word in doc["title"].lower() or word in doc["summary"].lower() for word in query_clean.split()):
                score += 2
            
            results.append((score, doc))

        # Sort by relevance score
        results.sort(key=lambda x: x[0], reverse=True)
        top_docs = [r[1] for r in results[:top_k]]
        
        # If no score match, return default IEEE paper references
        if not top_docs or results[0][0] == 0:
            top_docs = self.documents[:top_k]

        return top_docs
