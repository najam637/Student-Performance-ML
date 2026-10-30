"""Svm model for classification; includes its evaluation visualization."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

MODEL_NAME = "svm"
TASK = "classification"

def create_model():
    """Create a fresh estimator; preprocessing is added by the shared training pipeline."""
    return SVC(kernel="rbf", probability=True, random_state=42)

def parameter_grid(cfg):
    """Return this model's search grid from config (empty means no tuning)."""
    return {"model__C": cfg["hyperparameters"]["svm"]["C"], "model__gamma": cfg["hyperparameters"]["svm"]["gamma"]}

def plot_evaluation(y_true, y_pred, output_dir):
    """Save a model-specific held-out evaluation plot; does not alter training data."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    truth = np.asarray(y_true); pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if TASK == "regression":
        ax.scatter(truth, pred, alpha=.35, s=16)
        ax.plot([0, 3], [0, 3], linestyle="--")
        ax.set(xlabel="Actual grade code", ylabel="Predicted grade code", title=f"{MODEL_NAME.replace('_',' ').title()} — actual vs predicted")
        ax.set_xlim(-.1, 3.1); ax.set_ylim(-.1, 3.1)
    else:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(truth, pred, labels=[0, 1, 2, 3])
        ConfusionMatrixDisplay(cm, display_labels=["High", "Good", "Average", "Low"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"{MODEL_NAME.replace('_',' ').title()} — confusion matrix")
    fig.tight_layout()
    fig.savefig(output / f"{TASK}_{MODEL_NAME}_evaluation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
