"""Build sklearn preprocessing transformers; fitting happens on training data only."""
from pathlib import Path
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.logger import get_logger
from src.exception import CustomException
from src.utils.helpers import save_object

logger = get_logger(__name__)

def create_preprocessor(cfg, use_exam_score=True):
    try:
        numeric = list(cfg["features"]["numeric"])
        passthrough = list(cfg["features"]["passthrough"])
        nominal = list(cfg["features"]["nominal"])
        if not use_exam_score:
            numeric.remove("ExamScore")
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        nominal_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        return ColumnTransformer([("numeric", numeric_pipe, numeric), ("coded", "passthrough", passthrough), ("nominal", nominal_pipe, nominal)], remainder="drop", verbose_feature_names_out=False)
    except Exception as exc:
        raise CustomException("Could not create feature preprocessor", exc) from exc

class DataTransform:
    def run(self, train, cfg, root):
        try:
            x_train = train[cfg["features"]["all"]].copy()
            preprocessor = create_preprocessor(cfg, cfg.get("use_exam_score", True))
            preprocessor.fit(x_train)
            save_object(root / cfg["paths"]["preprocessor"], preprocessor)
            logger.info("Preprocessor fitted using training rows only")
            return preprocessor
        except Exception as exc:
            logger.exception("Data transformation failed")
            raise CustomException("Data transformation failed", exc) from exc
