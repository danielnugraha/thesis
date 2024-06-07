import argparse
from flwr_datasets.partitioner import (
    Partitioner,
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from typing import Dict
from dataloader.dataloader import Dataloader
from dataloader.multiclass_dataloader import CovertypeDataloader, LetterRecognitionDataloader
from dataloader.binary_dataloader import HiggsDataloader, RoadSafetyDataloader, MiniBooNEDataloader, EyeMovementsDataloader, JannisDataloader, CaliforniaDataloader
from dataloader.regression_dataloader import WineQualityDataloader, SuperconductDataloader, CpuActDataloader, DelaysZurichTransportDataloader, AllstateClaimsSeverityDataloader, HouseSalesDataloader, DiamondsDataloader

CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}

DATALOADER_MAPPING: Dict[str, Partitioner] = {
    "covertype": CovertypeDataloader,
    "letter_recognition": LetterRecognitionDataloader,
    "higgs": HiggsDataloader,
    "road_safety": RoadSafetyDataloader,
    "mini_boone": MiniBooNEDataloader,
    "eye_movements": EyeMovementsDataloader,
    "jannis": JannisDataloader,
    "california": CaliforniaDataloader,
    "wine_quality": WineQualityDataloader,
    "superconduct": SuperconductDataloader,
    "cpu_act": CpuActDataloader,
    "delays_zurich_transport": DelaysZurichTransportDataloader,
    "allstate_claims_severity": AllstateClaimsSeverityDataloader,
    "house_sales": HouseSalesDataloader,
    "diamonds": DiamondsDataloader,
}


def instantiate_dataloader(dataloader_type: str, partitioner: Partitioner) -> Dataloader:
    """Initialise dataloader based on selected dataloader type and additional
    keyword arguments."""
    dataloader_class = DATALOADER_MAPPING[dataloader_type]
    dataloader = dataloader_class(partitioner)
    return dataloader


def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type and number of
    partitions."""
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    return partitioner


# Hyper-parameters for xgboost training
NUM_LOCAL_ROUND = 1


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
        "--dataloader",
        default="covertype",
        type=str,
        choices=[
            "covertype", "letter_recognition", "higgs", "road_safety", "mini_boone",
            "eye_movements", "jannis", "california", "wine_quality", "superconduct",
            "cpu_act", "delays_zurich_transport", "allstate_claims_severity", 
            "house_sales", "diamonds"
        ],
        help="Dataloader types.",
    )
    parser.add_argument(
        "--partition-id",
        default=0,
        type=int,
        help="Partition ID used for the current client.",
    )
    parser.add_argument(
        "--sample-rate",
        default=0.1,
        type=float,
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
