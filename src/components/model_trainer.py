"""Train, tune, compare, evaluate, and persist every required model."""
from pathlib import Path
import importlib
import yaml
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split, cross_val_score
from src.components.data_transform import create_preprocessor
from src.pipeline.predict_pipeline import build_features
from src.logger import get_logger
from src.exception import CustomException
from src.utils.helpers import save_object, evaluate_models, write_json

logger = get_logger(__name__)
TARGET = "FinalGrade"

class ModelTrainer:
    def __init__(self, cfg, root): self.cfg, self.root = cfg, Path(root)

    def _pipeline(self, estimator, task, use_exam_score):
        return Pipeline([("preprocessor", create_preprocessor(self.cfg, use_exam_score)), ("model", estimator)])

    def _models(self, task):
        """Load each model from its own module, keeping algorithm code independently maintainable."""
        names = ("linear_regression", "decision_tree", "svm", "random_forest") if task == "regression" else ("decision_tree", "svm", "random_forest", "naive_bayes")
        return {
            name: (
                importlib.import_module(f"src.models.{task}.{name}").create_model(),
                importlib.import_module(f"src.models.{task}.{name}").parameter_grid(self.cfg) or None,
            )
            for name in names
        }

    def _train_task(self, train, test, task, use_exam_score):
        features = list(self.cfg["features"]["all"])
        if not use_exam_score: features.remove("ExamScore")
        # The same canonical feature builder is used by both training and the API.
        X_train_all = build_features(train[self.cfg["features"]["all"]])
        X_test_all = build_features(test[self.cfg["features"]["all"]])
        X_train, y_train = X_train_all[features].copy(), train[TARGET]
        X_test, y_test = X_test_all[features].copy(), test[TARGET]
        if task == "regression":
            cv = KFold(n_splits=5, shuffle=True, random_state=42); scoring = "r2"
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42); scoring = "f1_macro"
        results, fitted = [], {}
        for name, (estimator, grid) in self._models(task).items():
            try:
                pipe = self._pipeline(estimator, task, use_exam_score)
                cv_results = []
                if grid:
                    search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=1, refit=True, error_score="raise")
                    # Keep CV laptop-friendly by tuning on a deterministic sample taken
                    # ONLY from the training split. The required 5-fold splitter is still
                    # used. Non-SVM winners are refit on all training rows; RBF SVM/SVR
                    # remain fitted on the 1,000-row tuning sample because full-size RBF
                    # fitting is disproportionately expensive for this teaching project.
                    tune_x, tune_y = X_train, y_train
                    sample_limit = 500 if name == "svm" else 1200
                    if len(X_train) > sample_limit:
                        tune_x, _, tune_y, _ = train_test_split(
                            X_train, y_train, train_size=sample_limit, random_state=42,
                            stratify=y_train if task == "classification" else None
                        )
                    search.fit(tune_x, tune_y)
                    best_params = search.best_params_; cv_score = float(search.best_score_)
                    cv_results = [
                        {"params": params, "mean_test_score": float(score)}
                        for params, score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
                    ]
                    model = search.best_estimator_ if name == "svm" else clone(search.best_estimator_).set_params(**best_params).fit(X_train, y_train)
                else:
                    # Even untuned baselines receive the same training-only CV evaluation.
                    fold_scores = cross_val_score(pipe, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
                    cv_score = float(np.mean(fold_scores))
                    model = pipe.fit(X_train, y_train); best_params = {}
                pred = model.predict(X_test)
                metrics = evaluate_models(y_test, pred, task)
                model_module = importlib.import_module(f"src.models.{task}.{name}")
                plot_dir = self.root / "reports" / "figures" / ("with_exam_score" if use_exam_score else "without_exam_score")
                model_module.plot_evaluation(y_test, pred, plot_dir)
                row = {"task": task, "model": name, "use_exam_score": use_exam_score, "cv_best_score": cv_score, "best_params": best_params, "cv_results": cv_results, **metrics}
                results.append(row); fitted[name] = model
                suffix = "" if use_exam_score else "_no_exam"
                save_object(self.root / "models" / f"{task}{suffix}_{name}.pkl", model)
                logger.info("%s | exam_score=%s | %s", task, use_exam_score, row)
            except Exception as exc:
                logger.exception("Model failed: task=%s model=%s", task, name)
                raise CustomException(f"Training failed for {task}/{name}", exc) from exc
        # Select the default using training-only cross-validation, never held-out test metrics.
        best = max(results, key=lambda row: row["cv_best_score"] if row["cv_best_score"] is not None else float("-inf"))
        logger.info("Best %s model with ExamScore=%s: %s", task, use_exam_score, best["model"])
        return results, fitted, best

    def run(self, train, test):
        try:
            # Required default experiment, then a second leakage-aware experiment without ExamScore.
            reg_rows, reg_models, best_reg = self._train_task(train, test, "regression", True)
            clf_rows, clf_models, best_clf = self._train_task(train, test, "classification", True)
            no_exam_reg_rows, _, best_no_exam_reg = self._train_task(train, test, "regression", False)
            no_exam_clf_rows, _, best_no_exam_clf = self._train_task(train, test, "classification", False)
            save_object(self.root / "models/model_v1.pkl", reg_models[best_reg["model"]])
            save_object(self.root / "models/best_classifier.pkl", clf_models[best_clf["model"]])
            all_rows = reg_rows + clf_rows + no_exam_reg_rows + no_exam_clf_rows
            comparison = {
                "leakage_note": "FinalGrade is deterministically binned from ExamScore. Scores with ExamScore are leakage-inflated and must not be interpreted as independent prediction performance.",
                "with_exam_score": {"regression": reg_rows, "classification": clf_rows, "best_regressor": best_reg["model"], "best_classifier": best_clf["model"]},
                "without_exam_score": {"regression": no_exam_reg_rows, "classification": no_exam_clf_rows, "best_regressor": best_no_exam_reg["model"], "best_classifier": best_no_exam_clf["model"]},
            }
            metrics = {"r2": best_reg["r2"], "mae": best_reg["mae"], "rmse": best_reg["rmse"], "accuracy": best_clf["accuracy"], "macro_f1": best_clf["macro_f1"], "best_regressor": best_reg["model"], "best_classifier": best_clf["model"], "use_exam_score": True, "leakage_note": comparison["leakage_note"]}
            write_json(self.root / "models/metrics.json", metrics)
            write_json(self.root / "models/model_comparison.json", comparison)
            logger.info("Training complete. Default metrics: %s", metrics)
            return metrics, comparison
        except Exception as exc:
            logger.exception("Model training failed")
            raise CustomException("Model training failed", exc) from exc
