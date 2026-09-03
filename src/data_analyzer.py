import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def ensure_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def generate_default_raw_datasets():
    """Generates realistic initial raw source datasets if the user has not uploaded custom files yet."""
    ensure_directories()
    pm_path = os.path.join(RAW_DIR, "project_management_dataset.csv")
    skill_path = os.path.join(RAW_DIR, "skill_based_task_assignment.csv")

    if not os.path.exists(pm_path):
        pm_data = {
            "Project_Name": [f"Project_{i+1:03d}" for i in range(120)],
            "Project_Type": np.random.choice(["Web App", "Mobile App", "AI/ML Service", "Cloud Infrastructure", "Database Migration"], size=120),
            "Department": np.random.choice(["Software Engineering", "Data Science", "DevOps", "IT Operations"], size=120),
            "Project_Cost_USD": np.random.randint(5000, 50000, size=120),
            "Project_Benefit_Score": np.random.uniform(5.0, 9.9, size=120).round(2),
            "Priority": np.random.choice(["Low", "Medium", "High", "Critical"], size=120)
        }
        pd.DataFrame(pm_data).to_csv(pm_path, index=False)

    if not os.path.exists(skill_path):
        task_categories = ["Frontend", "Backend", "Database", "API Integration", "AI Model", "Testing/QA", "Deployment"]
        skills = {
            "Frontend": ["React", "Vue", "HTML/CSS", "JavaScript", "TypeScript"],
            "Backend": ["FastAPI", "Flask", "Node.js", "Django", "Python"],
            "Database": ["PostgreSQL", "SQLite", "MongoDB", "Redis"],
            "API Integration": ["REST API", "GraphQL", "OAuth2", "WebSockets"],
            "AI Model": ["PyTorch", "TensorFlow", "Scikit-Learn", "OpenAI API"],
            "Testing/QA": ["Pytest", "Jest", "Cypress", "Selenium"],
            "Deployment": ["Docker", "Kubernetes", "AWS", "GitHub Actions"]
        }
        records = []
        for i in range(350):
            cat = np.random.choice(task_categories)
            sk = np.random.choice(skills[cat])
            records.append({
                "Category": cat,
                "Sub_Category": f"{cat} Task Module {np.random.randint(1, 10)}",
                "Required_Skill": sk,
                "Complexity": np.random.choice(["Low", "Medium", "High"]),
                "Estimated_Effort_Hours": np.random.randint(4, 40)
            })
        pd.DataFrame(records).to_csv(skill_path, index=False)

def analyze_raw_datasets(ds1_path=None, ds2_path=None):
    """Analyses the two source datasets and decides synthetic data requirement."""
    ensure_directories()
    generate_default_raw_datasets()

    path1 = ds1_path if ds1_path and os.path.exists(ds1_path) else os.path.join(RAW_DIR, "project_management_dataset.csv")
    path2 = ds2_path if ds2_path and os.path.exists(ds2_path) else os.path.join(RAW_DIR, "skill_based_task_assignment.csv")

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    analysis = {
        "dataset_1": {
            "file": os.path.basename(path1),
            "rows": len(df1),
            "columns": list(df1.columns),
            "missing_values": int(df1.isnull().sum().sum()),
            "duplicates": int(df1.duplicated().sum())
        },
        "dataset_2": {
            "file": os.path.basename(path2),
            "rows": len(df2),
            "columns": list(df2.columns),
            "missing_values": int(df2.isnull().sum().sum()),
            "duplicates": int(df2.duplicated().sum())
        }
    }

    # Deterministic check for synthetic necessity
    execution_cols = ["actual_days", "is_delayed", "dependency_count", "actual_progress_pct"]
    has_execution_data = any(col in df1.columns.str.lower() for col in execution_cols) or any(col in df2.columns.str.lower() for col in execution_cols)

    if not has_execution_data:
        decision = "SYNTHETIC DATA REQUIRED"
        reason = ("The raw source datasets provide project categories and skill requirements, but lack "
                  "historical task execution metrics (actual durations, dependency links, member workloads, "
                  "and delay outcomes) required to evaluate risk and train prediction models.")
    else:
        decision = "SYNTHETIC DATA NOT REQUIRED"
        reason = "The source datasets already contain complete task execution and delay outcome variables."

    analysis["decision"] = decision
    analysis["reason"] = reason
    return analysis, df1, df2

def generate_master_dataset(df1, df2, analysis):
    """Generates inferential synthetic records via stratified sampling and merges into Master Dataset."""
    ensure_directories()
    records = []

    # Map complexity to estimated duration and delay probability
    complexity_map = {"Low": (1.5, 0.15), "Medium": (3.0, 0.30), "High": (5.0, 0.55)}

    categories = df2['Category'].unique() if 'Category' in df2.columns else ["Frontend", "Backend", "Database", "API Integration", "AI Model"]
    
    # Stratified generation across categories
    for i in range(500):
        cat = np.random.choice(categories)
        comp = np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2])
        base_days, delay_prob = complexity_map[comp]

        # Infer dependencies based on category order
        dep_count = 0
        if cat in ["Backend", "API Integration"]:
            dep_count = np.random.choice([1, 2])
        elif cat in ["Testing/QA", "Deployment"]:
            dep_count = np.random.choice([2, 3, 4])

        workload_ratio = np.random.uniform(0.6, 1.4)
        effective_delay_prob = min(0.9, delay_prob * (workload_ratio ** 1.2) * (1 + 0.1 * dep_count))
        is_delayed = 1 if np.random.rand() < effective_delay_prob else 0

        estimated_duration = round(base_days * np.random.uniform(0.9, 1.2), 1)
        actual_duration = round(estimated_duration * (np.random.uniform(1.2, 1.8) if is_delayed else np.random.uniform(0.8, 1.05)), 1)

        records.append({
            "Record_ID": f"REC_{i+1:04d}",
            "Project_Type": np.random.choice(df1['Project_Type'].unique()) if 'Project_Type' in df1.columns else "Web App",
            "Task_Category": cat,
            "Complexity": comp,
            "Required_Skill": np.random.choice(df2['Required_Skill'].unique()) if 'Required_Skill' in df2.columns else "Python",
            "Estimated_Duration_Days": estimated_duration,
            "Actual_Duration_Days": actual_duration,
            "Workload_Ratio": round(workload_ratio, 2),
            "Dependency_Count": dep_count,
            "Is_Delayed": is_delayed,
            "Risk_Level": "High" if is_delayed and actual_duration - estimated_duration > 2 else ("Medium" if is_delayed else "Low"),
            "Record_Type": "synthetic" if analysis["decision"] == "SYNTHETIC DATA REQUIRED" else "raw"
        })

    master_df = pd.DataFrame(records)
    master_path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    master_df.to_csv(master_path, index=False)
    return master_df

def train_and_evaluate_ml_pipeline():
    """80/10/10 Train/Validation/Test split and Random Forest Classifier training."""
    master_path = os.path.join(PROCESSED_DIR, "master_dataset.csv")
    if not os.path.exists(master_path):
        analysis, df1, df2 = analyze_raw_datasets()
        generate_master_dataset(df1, df2, analysis)

    df = pd.read_csv(master_path)
    
    # Feature Encoding
    df_encoded = pd.get_dummies(df[["Project_Type", "Task_Category", "Complexity", "Estimated_Duration_Days", "Workload_Ratio", "Dependency_Count"]], drop_first=True)
    X = df_encoded
    y = df["Is_Delayed"]

    # 80/10/10 Stratified Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Validation Set Evaluation
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    # Final Locked Test Set Evaluation
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset_total_records": len(df),
        "train_size": len(X_train),
        "validation_size": len(X_val),
        "test_size": len(X_test),
        "accuracy": round(accuracy_score(y_test, test_preds), 4),
        "precision": round(precision_score(y_test, test_preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, test_preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, test_preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, test_probs), 4),
        "feature_importances": dict(zip(X.columns, model.feature_importances_.round(4)))
    }

    # Save model and metadata
    ensure_directories()
    joblib.dump(model, os.path.join(MODELS_DIR, "delay_risk_model.pkl"))
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

if __name__ == "__main__":
    analysis, df1, df2 = analyze_raw_datasets()
    print("Dataset Analysis:", json.dumps(analysis, indent=2))
    master_df = generate_master_dataset(df1, df2, analysis)
    print("Master Dataset Created:", len(master_df), "records.")
    metrics = train_and_evaluate_ml_pipeline()
    print("ML Pipeline Evaluation Metrics:", json.dumps(metrics, indent=2))
