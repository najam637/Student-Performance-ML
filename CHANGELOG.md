# Project improvements

- Removed exact duplicate rows before the stratified train/test split and verified there is no identical-row overlap.
- Kept preprocessing inside each model Pipeline so transformations are fitted only on training folds.
- Added a separate Python module for every regression/classification model implementation, with its own estimator factory, hyperparameter grid, and held-out evaluation plot function.
- Added evaluation plots for every model in both the with-ExamScore and without-ExamScore experiments.
- Selected default models using training-only cross-validation rather than held-out test metrics.
- Added input bounds validation, `/health`, `/api/models`, `/api/metrics`, and `/api/model-comparison` endpoints.
- Fixed the duplicated regression model selector, added a classification model selector, exposed all comparison rows, and added a visualizations gallery.
- Made the UI display the ExamScore leakage warning beside the headline metrics and comparison table.
- Added tests for model modules, API validation, and the two leakage experiments.
