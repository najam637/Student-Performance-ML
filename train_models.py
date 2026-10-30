"""Backward-compatible training entry point: python train_models.py"""
from src.pipeline.train_pipeline import run_training

if __name__ == "__main__":
    metrics, _ = run_training()
    print("Training finished. Metrics:")
    for key, value in metrics.items(): print(f"{key}: {value}")
