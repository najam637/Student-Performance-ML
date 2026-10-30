# Folder Guide

Keep the project root open in JupyterLab's file browser. The folders are grouped by purpose:

- `app.py` — starts the website/API.
- `train_models.py` — retrains models and regenerates metrics/plots.
- `notebooks/` — two notebooks: exploratory data analysis and modeling.
- `data/raw/` — original dataset (keep this file).
- `src/components/` — data ingestion, preprocessing, and model training.
- `src/models/regression/` — one Python file per retained regression algorithm (Linear Regression, Decision Tree, SVM/SVR, Random Forest).
- `src/models/classification/` — one Python file per retained classification algorithm (Decision Tree, SVM, Random Forest, Naive Bayes).
- `src/pipeline/` — training and prediction workflows.
- `models/` — saved model artifacts and evaluation JSON used by the website. These are needed for immediate predictions and model selection.
- `reports/figures/` — generated model evaluation charts, split by experiment.
- `static/` — website HTML, CSS, and JavaScript.
- `tests/` — automated checks.
- `config/` — project configuration.

ANN and KNN source modules, saved artifacts, evaluation charts, UI choices, and notebook sections have been removed. The `.venv` environment and temporary logs are excluded to keep the Jupyter file browser tidy. Generated train/test splits are recreated by `python train_models.py`. Run training once after extraction so saved model artifacts and comparison metrics match the retained algorithms.
