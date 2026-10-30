"""Reusable artifact and evaluation helpers."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, confusion_matrix
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def save_object(path, obj, compress=3):
    try:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path, compress=compress)
        logger.info("Saved artifact: %s", path)
    except Exception as exc:
        raise CustomException("Could not save artifact", exc) from exc

def load_object(path):
    try:
        return joblib.load(path)
    except Exception as exc:
        raise CustomException(f"Could not load artifact {path}", exc) from exc

def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

def evaluate_models(y_true, y_pred, task):
    """Return the required held-out metrics for a regression or classification task."""
    if task == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return {"r2": float(r2_score(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred)), "rmse": float(np.sqrt(mse))}
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)), "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0,1,2,3]).tolist()}
