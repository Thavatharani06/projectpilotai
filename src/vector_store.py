import os
import json

class KnowledgeBaseRAG:
    def __init__(self):
        self.documents = [
            {
                "title": "FastAPI & Python Async Backend Guide",
                "skill": "Backend",
                "url": "https://fastapi.tiangolo.com/tutorial/",
                "summary": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints. Supports async def, OpenAPI schemas, and Pydantic validation."
            },
            {
                "title": "React.js Modern Hooks & State Management",
                "skill": "Frontend",
                "url": "https://react.dev/reference/react",
                "summary": "Best practices for React components using useState, useEffect, and custom hooks. Emphasizes clean UI state management, dynamic component rendering, and responsive design systems."
            },
            {
                "title": "SQLite Database Optimization & ORM Schemas",
                "skill": "Database",
                "url": "https://www.sqlite.org/docs.html",
                "summary": "SQLite index optimization, WAL (Write-Ahead Logging) mode, foreign key constraint enforcement, and lightweight ORM integrations with Python sqlite3/SQLAlchemy."
            },
            {
                "title": "RESTful API Integration & CORS Configuration",
                "skill": "API Integration",
                "url": "https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS",
                "summary": "Guidelines for integrating frontend fetch/axios calls with backend REST endpoints, handling CORS headers, Bearer Token authentication, and JSON error responses."
            },
            {
                "title": "Pytest Automation & Integration Testing",
                "skill": "Testing/QA",
                "url": "https://docs.pytest.org/en/stable/",
                "summary": "Automated testing framework for Python. Provides fixture management, assertions, test isolation, coverage metrics, and continuous integration pipeline automation."
            },
            {
                "title": "Docker Containerization & Multi-Stage Builds",
                "skill": "Deployment",
                "url": "https://docs.docker.com/get-started/",
                "summary": "Packaging web applications into Docker containers using multi-stage Dockerfiles. Configures lightweight production container images and environment variable security."
            },
            {
                "title": "Scikit-Learn Machine Learning Pipeline",
                "skill": "AI Model",
                "url": "https://scikit-learn.org/stable/user_guide.html",
                "summary": "Building supervised learning pipelines using RandomForestClassifier, StandardScaler, train_test_split, cross-validation, and metrics evaluation (Accuracy, F1, ROC-AUC)."
            }
        ]
        self.chroma_collection = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            client = chromadb.Client()
            self.chroma_collection = client.create_collection("projectpilot_docs")
            for idx, doc in enumerate(self.documents):
                self.chroma_collection.add(
                    documents=[f"{doc['title']}: {doc['summary']}"],
                    metadatas=[{"url": doc["url"], "skill": doc["skill"], "title": doc["title"]}],
                    ids=[f"doc_{idx}"]
                )
        except Exception:
            # Fallback to local keyword/text search if ChromaDB is unavailable
            self.chroma_collection = None

    def search_resources(self, query, skill=None, top_k=2):
        """Retrieves technical documentation relevant to the current project task or query."""
        results = []
        
        if self.chroma_collection:
            try:
                res = self.chroma_collection.query(query_texts=[query], n_results=top_k)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                for d, m in zip(docs, metas):
                    results.append({
                        "title": m["title"],
                        "url": m["url"],
                        "summary": d,
                        "skill": m["skill"]
                    })
                return results
            except Exception:
                pass

        # Fallback text matching
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_words = set((doc["title"] + " " + doc["summary"] + " " + doc["skill"]).lower().split())
            overlap = len(query_words.intersection(doc_words))
            if skill and doc["skill"].lower() == skill.lower():
                overlap += 3
            scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

if __name__ == "__main__":
    rag = KnowledgeBaseRAG()
    res = rag.search_resources("FastAPI backend integration", skill="Backend")
    print("Retrieved Resources:")
    for r in res:
        print(f" - [{r['title']}] ({r['url']}): {r['summary'][:80]}...")
