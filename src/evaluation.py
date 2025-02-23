from typing import Optional

from loguru import logger
from pathlib import Path
from src.data_loader import ValDataLoader
from src.metrics import Valid_Metrics_Type
from src.trainer import Trainer


def evaluate_model(
    trainer: Trainer,
    te_loader: ValDataLoader,
    te_loader_sh: ValDataLoader,
    val_metric: Valid_Metrics_Type,
    model_name: str,
    weights_path: Optional[Path] = None,
):
    if weights_path is not None:
        trainer.model.load_model(weights_path)

    logger.info(f"Evaluating {model_name} model")
    te_metric_results, _ = trainer.validate(te_loader, val_metric=val_metric)
    logger.info(f"Test {val_metric.name}: {te_metric_results:.5f}")

    te_sh_metric_results, _ = trainer.validate(te_loader_sh, val_metric=val_metric)
    logger.info(f"Test {val_metric.name} on shared users only: {te_sh_metric_results:.5f}")
