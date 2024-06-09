import argparse
from flwr_datasets.partitioner import (
    Partitioner,
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)

from subsampling.goss import GOSS
from subsampling.mvs import MVS
from subsampling.random import Random

from typing import Dict
from dataloader import Dataloader, CovertypeDataloader, HelenaDataloader, DionisDataloader, HiggsDataloader, RoadSafetyDataloader, JannisDataloader, WineQualityDataloader, AllstateClaimsSeverityDataloader, HouseSalesDataloader, DiamondsDataloader

PARTITIONER_MAPPING = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}

DATALOADER_MAPPING = {
    "covertype": CovertypeDataloader,
    "higgs": HiggsDataloader,
    "road_safety": RoadSafetyDataloader,
    "jannis": JannisDataloader,
    "wine_quality": WineQualityDataloader,
    "allstate_claims_severity": AllstateClaimsSeverityDataloader,
    "house_sales": HouseSalesDataloader,
    "diamonds": DiamondsDataloader,
    "helena": HelenaDataloader,
    "dionis": DionisDataloader,
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
    partitioner = PARTITIONER_MAPPING[partitioner_type](
        num_partitions=num_partitions
    )
    return partitioner


def instantiate_sampling_method(sampling_method: str, objective, sample_rate: float):
    if sampling_method == "mvs":
        return MVS(objective, sample_rate=sample_rate)
    if sampling_method == "goss":
        return GOSS(objective, sample_rate/2, sample_rate/2)
    if sampling_method == "random":
        return Random(objective, sample_rate)


# Hyper-parameters for xgboost training
NUM_LOCAL_ROUND = 1


def generic_args_parser():
    parser = argparse.ArgumentParser()

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
            "covertype", "helena", "dionis", "higgs", "road_safety", "jannis", 
            "wine_quality", "allstate_claims_severity", "house_sales", "diamonds"
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
        "--sampling-method",
        default="mvs",
        type=str,
        choices=["mvs", "goss", "random", "native"],
        help="Subsampling methods for running XGBoost.",
    )

    args = parser.parse_args()
    return args
