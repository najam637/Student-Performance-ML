"""Run ingestion, transformation, and model training in the required order."""
from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransform
from src.components.model_trainer import ModelTrainer
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def run_training():
    try:
        train, test, cfg, root = DataIngestion().run()
        DataTransform().run(train, cfg, root)
        metrics, comparison = ModelTrainer(cfg, root).run(train, test)
        return metrics, comparison
    except Exception as exc:
        logger.exception("Training pipeline failed")
        raise CustomException("Training pipeline failed", exc) from exc

if __name__ == "__main__":
    print(run_training()[0])
