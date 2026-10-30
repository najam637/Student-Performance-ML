"""Artifact loading, prediction range, and model quality smoke tests."""
from pathlib import Path
import json
import pandas as pd
import pytest
from src.pipeline.predict_pipeline import PredictPipeline, FEATURE_COLUMNS
ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not (ROOT / "models/model_v1.pkl").exists(), reason="Run python train_models.py first")
def test_model_loading_and_regression_range():
    row = pd.read_csv(ROOT / "data/raw/student_performance.csv")[FEATURE_COLUMNS].iloc[[0]]
    result = PredictPipeline(ROOT / "models").predict(row, "regression")
    assert 0 <= result["predicted_final_grade_code"] <= 3

@pytest.mark.skipif(not (ROOT / "models/metrics.json").exists(), reason="Run python train_models.py first")
def test_default_metrics_meet_assignment_thresholds():
    metrics = json.loads((ROOT / "models/metrics.json").read_text())
    assert metrics["r2"] >= 0.8
    assert metrics["accuracy"] >= 0.8

def test_model_modules_are_separate_and_have_visualization_hooks():
    from src.models.regression import linear_regression, random_forest
    from src.models.classification import naive_bayes
    assert linear_regression.create_model() is not None
    assert random_forest.create_model() is not None
    assert naive_bayes.create_model() is not None
    assert callable(linear_regression.plot_evaluation)
    assert callable(random_forest.plot_evaluation)
    assert callable(naive_bayes.plot_evaluation)

def test_api_rejects_out_of_domain_values():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    payload = {c: 1 for c in FEATURE_COLUMNS}
    payload.update({"StudyHours": 999, "Attendance": 80, "Age": 20, "OnlineCourses": 5,
                    "AssignmentCompletion": 80, "ExamScore": 80, "model": "best"})
    response = client.post("/predict/classification", json=payload)
    assert response.status_code == 422

def test_model_comparison_contains_both_leakage_experiments():
    path = ROOT / "models/model_comparison.json"
    if not path.exists():
        pytest.skip("Run python train_models.py first")
    comparison = json.loads(path.read_text())
    assert set(comparison) >= {"with_exam_score", "without_exam_score", "leakage_note"}
    assert len(comparison["with_exam_score"]["regression"]) == 4
    assert len(comparison["without_exam_score"]["classification"]) == 4
