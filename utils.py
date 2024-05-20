import numpy as np
import xgboost as xgb
import argparse
from thesis_dataset import ThesisDataset
from datasets import Dataset, DatasetDict
from typing import Union
import math

params = {
    "objective": "reg:squarederror",
    "eta": 0.05,  # Learning rate
    "max_depth": 8,
    "eval_metric": "rmse",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}

multi_params = {
    "objective": "multi:softmax",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
    "num_class": 7,
}

dataset_params = {
    ThesisDataset.IRIS: {"objective": "multi:softmax"},
    ThesisDataset.HIGGS: {"objective": "binary:logistic"},
}

def iris_preprocess(data: Union[Dataset, DatasetDict]):
    x_dict = data.with_format("np", ["petal_length", "petal_width", "sepal_length", "sepal_width"])[:]
    x_arrays = list(x_dict.values())
    x = np.stack(x_arrays, axis=1)
    y = data["species"]
    return x, y

def higgs_preprocess(data: Union[Dataset, DatasetDict]):
    x = data["inputs"]
    y = data["label"]
    return x, y

dataset_preprocess = {
    ThesisDataset.IRIS: iris_preprocess,
    ThesisDataset.HIGGS: higgs_preprocess
}

def get_params(dataset):
    params = {
        "eta": 0.1,  # Learning rate
        "max_depth": 8,
        "eval_metric": "auc",
        "nthread": 16,
        "num_parallel_tree": 1,
        "subsample": 1,
        "tree_method": "hist",
        'num_class': 3,
    }
    params.update(dataset_params.get(dataset, {}))
    return params


# Hyper-parameters for xgboost training
NUM_LOCAL_ROUND = 1
BST_PARAMS = {
    "objective": "binary:logistic",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


def client_args_parser():
    """Parse arguments to define experimental settings on client side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )
    parser.add_argument(
        "--num-partitions", default=10, type=int, help="Number of partitions."
    )
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--partition-id",
        default=0,
        type=int,
        help="Partition ID used for the current client.",
    )
    parser.add_argument(
        "--sampling-rate",
        default=0.1,
        type=int,
        help="Sampling rate .",
    )

    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct evaluation on centralised test set (True), or on hold-out data (False).",
    )
    parser.add_argument(
        "--scaled-lr",
        action="store_true",
        help="Perform scaled learning rate based on the number of clients (True).",
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Conduct evaluation on centralised test set (True), or on hold-out data (False).",
    )

    args = parser.parse_args()
    return args


def server_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )
    parser.add_argument(
        "--pool-size", default=2, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num-rounds", default=5, type=int, help="Number of FL rounds."
    )
    parser.add_argument(
        "--num-clients-per-round",
        default=2,
        type=int,
        help="Number of clients participate in training each round.",
    )
    parser.add_argument(
        "--num-evaluate-clients",
        default=2,
        type=int,
        help="Number of clients selected for evaluation.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args


def sim_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-method",
        default="bagging",
        type=str,
        choices=["bagging", "cyclic"],
        help="Training methods selected from bagging aggregation or cyclic training.",
    )

    # Server side
    parser.add_argument(
        "--pool-size", default=5, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num-rounds", default=30, type=int, help="Number of FL rounds."
    )
    parser.add_argument(
        "--num-clients-per-round",
        default=5,
        type=int,
        help="Number of clients participate in training each round.",
    )
    parser.add_argument(
        "--num-evaluate-clients",
        default=5,
        type=int,
        help="Number of clients selected for evaluation.",
    )
    parser.add_argument(
        "--centralised-eval",
        action="store_true",
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )
    parser.add_argument(
        "--num-cpus-per-client",
        default=2,
        type=int,
        help="Number of CPUs used for per client.",
    )

    # Client side
    parser.add_argument(
        "--partitioner-type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test-fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting.",
    )
    parser.add_argument(
        "--centralised-eval-client",
        action="store_true",
        help="Conduct evaluation on centralised test set (True), or on hold-out data (False).",
    )
    parser.add_argument(
        "--scaled-lr",
        action="store_true",
        help="Perform scaled learning rate based on the number of clients (True).",
    )

    args = parser.parse_args()
    return args
