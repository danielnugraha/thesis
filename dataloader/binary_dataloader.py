from dataloader.dataloader import Dataloader, train_test_split
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.resplitter import Resplitter
from typing import Optional, Union, Dict, Tuple
from datasets import Dataset, DatasetDict
import xgboost as xgb
import numpy as np
from objective import binary_obj

class HiggsDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.fds = FederatedDataset(
            dataset="jxie/higgs",
            partitioners={"train": partitioner},
            resplitter=resplitter,
        )
        self.resplitter = resplitter

    def get_train_dmatrix(self, node_id: Optional[int] = None) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.fds.load_full("train")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")
        
        partition.set_format("numpy")
        train_data, _ = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=train_data), train_data.shape[0]
    
    def get_test_dmatrix(self, node_id: Optional[int]) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.fds.load_full("test")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")
        
        partition.set_format("numpy")
        _, test_data = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=test_data), test_data.shape[0]

    def get_objective(self):
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return 2

    def set_partitioner(self, partitioner: Partitioner) -> None:
        self.fds = FederatedDataset(
            dataset="jxie/higgs",
            partitioners={"train": partitioner},
            resplitter=self.resplitter,
        )

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x = data["inputs"]
        y = data["label"]
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    