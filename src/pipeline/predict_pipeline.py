"""Shared feature validation and artifact-backed prediction classes."""
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.exception import CustomException
from src.utils.helpers import load_object

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
FEATURE_COLUMNS = CONFIG["features"]["all"]
CLASS_NAMES = {0: "High", 1: "Good", 2: "Average", 3: "Low"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the exact 15 input fields and return them in the canonical order."""
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("build_features expects a pandas DataFrame")
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in FEATURE_COLUMNS]
        if missing or extra or list(df.columns) != FEATURE_COLUMNS:
            raise ValueError(
                f"Feature columns/order mismatch. Expected={FEATURE_COLUMNS}; "
                f"received={list(df.columns)}; missing={missing}; extra={extra}"
            )
        frame = df.copy()
        if frame.isna().any().any():
            raise ValueError("Prediction inputs must not contain null values")
        for column in FEATURE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame
    except Exception as exc:
        raise CustomException("Feature validation failed", exc) from exc

class CustomData:
    """Named holder for the 15 values submitted by the web form."""
    def __init__(self, **kwargs):
        missing = [name for name in FEATURE_COLUMNS if name not in kwargs]
        if missing:
            raise ValueError(f"Missing input values: {missing}")
        self.values = {name: kwargs[name] for name in FEATURE_COLUMNS}

    def get_data_as_dataframe(self):
        return build_features(pd.DataFrame([self.values]))

class PredictPipeline:
    """Load trained model pipelines and perform safe, clipped predictions."""
    REGRESSION_MODELS = {"linear_regression", "decision_tree", "svm", "random_forest"}
    CLASSIFICATION_MODELS = {"decision_tree", "svm", "random_forest", "naive_bayes"}

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else ROOT / "models"

    def available_models(self, task):
        allowed = self.REGRESSION_MODELS if task == "regression" else self.CLASSIFICATION_MODELS
        return sorted(name for name in allowed if (self.models_dir / f"{task}_{name}.pkl").exists())

    def predict(self, data, task="regression", model_name=None):
        try:
            frame = data if isinstance(data, pd.DataFrame) else data.get_data_as_dataframe()
            frame = build_features(frame)
            best_file = "model_v1.pkl" if task == "regression" else "best_classifier.pkl"
            chosen = model_name or ("best" if not (self.models_dir / best_file).exists() else None)
            if chosen and chosen != "best":
                if chosen not in (self.REGRESSION_MODELS if task == "regression" else self.CLASSIFICATION_MODELS):
                    raise ValueError(f"Model {chosen!r} is not available for {task}")
                path = self.models_dir / f"{task}_{chosen}.pkl"
            else:
                path = self.models_dir / best_file
            model = load_object(path)
            prediction = model.predict(frame)[0]
            if task == "regression":
                return {"predicted_final_grade_code": round(float(np.clip(prediction, 0, 3)), 2), "model_used": path.stem}
            code = int(np.clip(round(float(prediction)), 0, 3))
            return {"predicted_grade_category": CLASS_NAMES[code], "model_used": path.stem}
        except Exception as exc:
            logger.exception("Prediction failed")
            raise CustomException("Prediction failed", exc) from exc
