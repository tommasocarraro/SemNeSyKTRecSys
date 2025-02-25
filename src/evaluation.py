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
    log_string: str,
    weights_path: Optional[Path] = None,
):
    """
    Evaluate the model attached to the given trainer. If the weights path is given, said weights are loaded before
    evaluation.

    :param trainer: The trainer whose model is to be evaluated
    :param te_loader: The test data loader
    :param te_loader_sh: The test data loader containing just ratings from shared users
    :param val_metric: The validation metric
    :param log_string: The log string to show before the evaluation
    :param weights_path: The path to the weights file
    """
    if weights_path is not None:
        trainer.model.load_model(weights_path)

    logger.info(log_string)
    te_metric_results, _ = trainer.validate(te_loader, val_metric=val_metric)
    logger.info(f"Test {val_metric.value}: {te_metric_results:.5f}")

    te_sh_metric_results, _ = trainer.validate(te_loader_sh, val_metric=val_metric)
    logger.info(f"Test {val_metric.value} on shared users only: {te_sh_metric_results:.5f}")
