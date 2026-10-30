"""FastAPI service for dataset exploration and student-grade prediction."""
from pathlib import Path
import json
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.pipeline.predict_pipeline import PredictPipeline, FEATURE_COLUMNS, CLASS_NAMES, build_features
from src.logger import get_logger

ROOT = Path(__file__).resolve().parent
logger = get_logger(__name__)
CFG = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
DATA_PATH = ROOT / CFG["paths"]["raw_data"]
MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
app = FastAPI(title="Student Performance Predictor", version="2.0.0", description="A modular ML app for regression and grade-category classification.")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/reports", StaticFiles(directory=ROOT / "reports", check_dir=False), name="reports")
predictor = PredictPipeline(MODELS_DIR)

class StudentInput(BaseModel):
    StudyHours: float = Field(ge=5, le=44)
    Attendance: float = Field(ge=60, le=100)
    Resources: int = Field(ge=0, le=2)
    Extracurricular: int = Field(ge=0, le=1)
    Motivation: int = Field(ge=0, le=2)
    Internet: int = Field(ge=0, le=1)
    Gender: int = Field(ge=0, le=1)
    Age: int = Field(ge=18, le=29)
    LearningStyle: int = Field(ge=0, le=3)
    OnlineCourses: int = Field(ge=0, le=20)
    Discussions: int = Field(ge=0, le=1)
    AssignmentCompletion: int = Field(ge=50, le=100)
    ExamScore: float = Field(ge=40, le=100)
    EduTech: int = Field(ge=0, le=1)
    StressLevel: int = Field(ge=0, le=2)
    model: str | None = None

    def feature_frame(self):
        values = self.model_dump(exclude={"model"})
        return build_features(pd.DataFrame([values]))

def read_json(path, fallback):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return fallback

@app.get("/")
def home(): return FileResponse(ROOT / "static/index.html")

@app.get("/health")
def health():
    models_ready = (MODELS_DIR / "model_v1.pkl").exists() and (MODELS_DIR / "best_classifier.pkl").exists()
    return {"status": "ok" if models_ready else "degraded", "models_ready": models_ready}

@app.get("/api/models")
def api_models():
    """List the separately trained models available for each prediction task."""
    return {
        "regression": predictor.available_models("regression"),
        "classification": predictor.available_models("classification"),
        "best_regressor": read_json(MODELS_DIR / "metrics.json", {}).get("best_regressor"),
        "best_classifier": read_json(MODELS_DIR / "metrics.json", {}).get("best_classifier"),
    }

@app.get("/dataset/info")
def dataset_info():
    try:
        df = pd.read_csv(DATA_PATH)
        metrics = read_json(MODELS_DIR / "metrics.json", {})
        features = {}
        for col in FEATURE_COLUMNS:
            series = df[col]
            features[col] = {"min": int(series.min()), "max": int(series.max()), "default": int(series.median())}
        return {"rows": int(len(df)), "columns": int(df.shape[1]), "features": features,
                "regression": {"r2": metrics.get("r2", 0), "mae": metrics.get("mae", 0), "rmse": metrics.get("rmse", 0), "best_model": metrics.get("best_regressor", "not trained")},
                "classification": {"accuracy": metrics.get("accuracy", 0), "macro_f1": metrics.get("macro_f1", 0), "best_model": metrics.get("best_classifier", "not trained")},
                "leakage_note": metrics.get("leakage_note", "Train models before interpreting metrics.")}
    except Exception as exc:
        logger.exception("Could not load dataset information")
        raise HTTPException(status_code=500, detail="Could not read dataset information") from exc

@app.get("/dataset/sample")
def dataset_sample():
    try:
        row = pd.read_csv(DATA_PATH).iloc[0]
        return {col: int(row[col]) for col in FEATURE_COLUMNS}
    except Exception as exc: raise HTTPException(status_code=500, detail="Could not load a sample row") from exc

@app.get("/api/metrics")
def api_metrics(): return read_json(MODELS_DIR / "metrics.json", {"detail": "Run python train_models.py first."})

@app.get("/api/model-comparison")
def api_model_comparison(): return read_json(MODELS_DIR / "model_comparison.json", {"detail": "Run python train_models.py first."})

@app.post("/predict/regression")
def predict_regression(student: StudentInput):
    try:
        selected = student.model if student.model not in (None, "", "best") else None
        result = predictor.predict(student.feature_frame(), task="regression", model_name=selected)
        return {"predicted_final_grade_code": result["predicted_final_grade_code"], "model_used": result["model_used"]}
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/predict/classification")
def predict_classification(student: StudentInput):
    try:
        selected = student.model if student.model not in (None, "", "best") else None
        result = predictor.predict(student.feature_frame(), task="classification", model_name=selected)
        return {"predicted_grade_category": result["predicted_grade_category"], "model_used": result["model_used"]}
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
