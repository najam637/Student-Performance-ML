"""Read the raw CSV, remove duplicates, stratify split, and persist split files."""
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)
TARGET = "FinalGrade"

def load_config():
    root = Path(__file__).resolve().parents[2]
    return root, yaml.safe_load((root / "config/config.yaml").read_text(encoding="utf-8"))

class DataIngestion:
    def run(self):
        try:
            root, cfg = load_config()
            raw_path = root / cfg["paths"]["raw_data"]
            data = pd.read_csv(raw_path)
            expected = cfg["features"]["all"] + [TARGET]
            missing = [c for c in expected if c not in data.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            data = data[expected]
            if data.isna().any().any():
                raise ValueError("Raw dataset contains missing values; resolve them before training.")
            before = len(data)
            data = data.drop_duplicates().reset_index(drop=True)
            logger.info("Rows before/after duplicate removal: %s/%s", before, len(data))
            train, test = train_test_split(data, test_size=cfg["split"]["test_size"], random_state=cfg["split"]["random_state"], stratify=data[TARGET])
            out = root / cfg["paths"]["processed_dir"]
            out.mkdir(parents=True, exist_ok=True)
            train.to_csv(out / "train.csv", index=False)
            test.to_csv(out / "test.csv", index=False)
            # A row fingerprint audit makes the no-overlap guarantee explicit.
            train_fps = set(pd.util.hash_pandas_object(train, index=False).astype(str))
            test_fps = set(pd.util.hash_pandas_object(test, index=False).astype(str))
            if train_fps & test_fps:
                raise RuntimeError("Duplicate row overlap detected between train and test.")
            logger.info("Saved stratified split: train=%d test=%d", len(train), len(test))
            return train, test, cfg, root
        except Exception as exc:
            logger.exception("Data ingestion failed")
            raise CustomException("Data ingestion failed", exc) from exc
