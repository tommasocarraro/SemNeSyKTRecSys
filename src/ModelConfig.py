from pathlib import Path
from typing import Any, Literal, Union, get_args
import math

from loguru import logger

from src.utils import str_is_float

Valid_Metrics_Type = Literal["mse", "rmse", "fbeta", "acc", "auc"]
# because type hints and linters on Python are a joke
valid_metrics = list(get_args(Valid_Metrics_Type))


class ModelConfig:
    def __init__(self, config_json: dict[str, Any]):
        try:
            # dataset config
            datasets_config = config_json["datasets_config"]
            self.src_ratings_path = Path(datasets_config["src_ratings_path"])
            self.tgt_ratings_path = Path(datasets_config["tgt_ratings_path"])
            self.paths_file_path = Path(datasets_config["paths_file_path"])

            # shared config
            shared_config = config_json["shared_config"]
            self.epochs = int(shared_config["epochs"])
            self.early_stopping_patience = int(shared_config["early_stopping_patience"])
            self.early_stopping_loss = bool(shared_config["early_stopping_loss"])
            self.val_metric = shared_config["val_metric"]
            check_metrics(self.val_metric)
            self.seed = int(shared_config["seed"])

            # train config
            train_config = config_json["train_config"]
            self.n_factors = int(train_config["n_factors"])
            self.learning_rate = float(train_config["learning_rate"])
            self.weight_decay = float(train_config["weight_decay"])
            self.batch_size = int(train_config["batch_size"])
            self.save_path = config_json["save_path"]

            # tune config
            tune_config = config_json["tune_config"]
            self.sweep_config = {
                "method": tune_config["method"],
                "metric": {
                    "goal": tune_config["metric"]["goal"],
                    "name": tune_config["metric"]["name"],
                },
                "parameters": {
                    "n_factors": {
                        "values": tune_config["parameters"]["n_factors_range"]
                    },
                    "learning_rate": {
                        "min": math.log(tune_config["parameters"]["learning_rate"]["min"]),
                        "max": math.log(tune_config["parameters"]["learning_rate"]["max"]),
                        "distribution": tune_config["parameters"]["learning_rate"]["distribution"]
                    },
                    "weight_decay": {
                        "min": math.log(tune_config["parameters"]["weight_decay"]["min"]),
                        "max": math.log(tune_config["parameters"]["weight_decay"]["max"]),
                        "distribution": tune_config["parameters"]["weight_decay"]["distribution"]
                    },
                    "batch_size": {
                        "values": tune_config["parameters"]["batch_size_range"]
                    },
                },
            }
            self.entity_name = tune_config["entity_name"]
            self.exp_name = tune_config["exp_name"]
            self.bayesian_run_count = tune_config["bayesian_run_count"]
            self.sweep_id = tune_config["sweep_id"]
        except (KeyError, ValueError) as e:
            logger.error(e)
            exit(1)

    def get_train_config(self):
        return f"n_factors: {self.n_factors}, learning_rate: {self.learning_rate}, weight_decay: {self.weight_decay}, batch_size: {self.batch_size}"


def check_metrics(metrics: Union[str, list[str]]):
    """
    Check if the given list of metrics' names is correct.

    :param metrics: list of str containing the name of some metrics
    """
    err_msg = f"Some of the given metrics are not valid. The accepted metrics are {valid_metrics}"
    if isinstance(metrics, str):
        metrics = [metrics]
    assert all(
        [isinstance(m, str) for m in metrics]
    ), "The metrics must be represented as strings"
    assert all([m in valid_metrics for m in metrics if "-" not in m]), err_msg
    assert all(
        [
            m.split("-")[0] in valid_metrics and str_is_float(m.split("-")[1])
            for m in metrics
            if "-" in m
        ]
    ), err_msg
