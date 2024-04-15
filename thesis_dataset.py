from flwr_datasets import FederatedDataset
import xgboost as xgb
from typing import Union
from datasets import Dataset, DatasetDict, concatenate_datasets
from enum import Enum
import numpy as np
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)


class ThesisDataset(Enum):
    IRIS = "hitorilabs/iris"
    HIGGS = "jxie/higgs"
    ABALONE = "mstz/abalone"
    COVERTYPE = "mstz/covertype"


CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type and number of
    partitions."""
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    return partitioner


def create_centralized_dataset(dataset: str) -> tuple[xgb.core.DMatrix, xgb.core.DMatrix]:
    fds = FederatedDataset(dataset=dataset, partitioners={"train": 1, "test": 1})
    partition = fds.load_full("train")
    partition.set_format("numpy")
    train_data, test_data = train_test_split(
        partition, test_fraction=0.2, seed=42
    )
    return (transform_dataset_to_dmatrix(train_data), transform_dataset_to_dmatrix(test_data))


def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x, y = separate_xy(data)
    new_data = xgb.DMatrix(x, label=y)
    print(new_data.feature_names)
    return new_data


def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    return partition_train, partition_test


def separate_xy(data: Union[Dataset, DatasetDict]):
    """Return outputs of x (data) and y (labels) ."""
    x_dict = data.with_format("np", ["petal_length", "petal_width", "sepal_length", "sepal_width"])[:]
    x_arrays = list(x_dict.values())
    x = np.stack(x_arrays, axis=1)
    y = data["species"]
    return x, y


def resplit(dataset: DatasetDict) -> DatasetDict:
    """Increase the quantity of centralised test samples from 500K to 1M."""
    return DatasetDict(
        {
            "train": dataset["train"].select(
                range(0, dataset["train"].num_rows - 500_000)
            ),
            "test": concatenate_datasets(
                [
                    dataset["train"].select(
                        range(
                            dataset["train"].num_rows - 500_000,
                            dataset["train"].num_rows,
                        )
                    ),
                    dataset["test"],
                ]
            ),
        }
    )
