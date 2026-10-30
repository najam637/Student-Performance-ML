"""Dataset and feature contract tests."""
from pathlib import Path
import pandas as pd
import pytest
from src.pipeline.predict_pipeline import FEATURE_COLUMNS, build_features
ROOT = Path(__file__).resolve().parents[1]

def test_raw_dataset_has_required_columns_and_no_nulls():
    data = pd.read_csv(ROOT / "data/raw/student_performance.csv")
    assert set(FEATURE_COLUMNS + ["FinalGrade"]).issubset(data.columns)
    assert not data[FEATURE_COLUMNS + ["FinalGrade"]].isna().any().any()

def test_build_features_returns_15_columns_in_canonical_order():
    sample = {c: 1 for c in FEATURE_COLUMNS}
    frame = build_features(pd.DataFrame([sample]))
    assert frame.shape == (1, 15)
    assert frame.columns.tolist() == FEATURE_COLUMNS

def test_duplicate_rows_are_removed_before_split():
    data = pd.read_csv(ROOT / "data/raw/student_performance.csv")
    assert len(data.drop_duplicates()) < len(data)
    train_path, test_path = ROOT / "data/processed/train.csv", ROOT / "data/processed/test.csv"
    if train_path.exists() and test_path.exists():
        train, test = pd.read_csv(train_path), pd.read_csv(test_path)
        h1 = set(pd.util.hash_pandas_object(train, index=False).astype(str))
        h2 = set(pd.util.hash_pandas_object(test, index=False).astype(str))
        assert h1.isdisjoint(h2)


def test_build_features_rejects_wrong_column_order():
    sample = {c: 1 for c in reversed(FEATURE_COLUMNS)}
    with pytest.raises(Exception):
        build_features(pd.DataFrame([sample]))

def test_fitted_preprocessor_output_has_no_nulls():
    import numpy as np
    import pytest
    preprocessor_path = ROOT / "models/preprocessor.pkl"
    if not preprocessor_path.exists():
        pytest.skip("Run python train_models.py first")
    import joblib
    train_path = ROOT / "data/processed/train.csv"
    if not train_path.exists():
        pytest.skip("Run python train_models.py first")
    train = pd.read_csv(train_path)
    preprocessor = joblib.load(preprocessor_path)
    transformed = preprocessor.transform(train[FEATURE_COLUMNS])
    values = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    assert np.isfinite(values).all()
