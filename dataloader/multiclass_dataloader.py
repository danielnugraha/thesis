from dataloader.dataloader import Dataloader, train_test_split
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.resplitter import Resplitter
from typing import Optional, Union, Dict, Tuple
from datasets import Dataset, DatasetDict
import xgboost as xgb
import numpy as np
from objective import softprob_obj

class IrisDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.fds = FederatedDataset(
            dataset="hitorilabs/iris",
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
            partition = self.fds.load_full("train")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")
        
        partition.set_format("numpy")
        _, test_data = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=test_data), test_data.shape[0]

    def get_objective(self):
        return softprob_obj
    
    def get_params(self):
        return {
            "objective": "multi:softmax",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
            "num_class": 3,
        }

    def get_num_classes(self):
        return 3

    def set_partitioner(self, partitioner: Partitioner) -> None:
        self.fds = FederatedDataset(
            dataset="hitorilabs/iris",
            partitioners={"train": partitioner},
            resplitter=self.resplitter,
        )

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ["petal_length", "petal_width", "sepal_length", "sepal_width"])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data["species"]
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    

class CovertypeDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.fds = FederatedDataset(
            dataset="mstz/covertype",
            subset="covertype",
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
        train_data, test_data = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=train_data), train_data.shape[0]
    
    def get_test_dmatrix(self, node_id: Optional[int]) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.fds.load_full("train")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")
        
        partition.set_format("numpy")
        _, test_data = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=test_data), test_data.shape[0]

    def get_objective(self):
        return softprob_obj
    
    def get_params(self):
        return {
            "objective": "multi:softprob",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
            "num_class": 7,
        }

    def get_num_classes(self):
        return 7

    def set_partitioner(self, partitioner: Partitioner) -> None:
        self.fds = FederatedDataset(
            dataset="mstz/covertype",
            subset="covertype",
            partitioners={"train": partitioner},
            resplitter=self.resplitter,
        )

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways', 'hillshade_9am', 'hillshade_noon', 'hillshade_3pm', 'horizontal_distance_to_fire_points', 'wilderness_area_id_0', 'wilderness_area_id_1', 'wilderness_area_id_2', 'wilderness_area_id_3', 'soil_type_id_0', 'soil_type_id_1', 'soil_type_id_2', 'soil_type_id_3', 'soil_type_id_4', 'soil_type_id_5', 'soil_type_id_6', 'soil_type_id_7', 'soil_type_id_8', 'soil_type_id_9', 'soil_type_id_10', 'soil_type_id_11', 'soil_type_id_12', 'soil_type_id_13', 'soil_type_id_14', 'soil_type_id_15', 'soil_type_id_16', 'soil_type_id_17', 'soil_type_id_18', 'soil_type_id_19', 'soil_type_id_20', 'soil_type_id_21', 'soil_type_id_22', 'soil_type_id_23', 'soil_type_id_24', 'soil_type_id_25', 'soil_type_id_26', 'soil_type_id_27', 'soil_type_id_28', 'soil_type_id_29', 'soil_type_id_30', 'soil_type_id_31', 'soil_type_id_32', 'soil_type_id_33', 'soil_type_id_34', 'soil_type_id_35', 'soil_type_id_36', 'soil_type_id_37', 'soil_type_id_38', 'soil_type_id_39'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['cover_type']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class LetterRecognitionDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.fds = FederatedDataset(
            dataset="wwydmanski/tabular-letter-recognition",
            partitioners={"train": partitioner, "test": partitioner},
            resplitter=resplitter,
        )
        self.resplitter = resplitter

    def get_train_dmatrix(self, node_id: Optional[int] = None) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.fds.load_full("train")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")
        
        partition.set_format("numpy")
        
        return self._transform_dataset_to_dmatrix(data=partition), partition.shape[0]
    
    def get_test_dmatrix(self, node_id: Optional[int]) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.fds.load_full("test")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="test")
        
        partition.set_format("numpy")
        
        return self._transform_dataset_to_dmatrix(data=partition), partition.shape[0]

    def get_objective(self):
        return softprob_obj
    
    def get_params(self):
        return {
            "objective": "multi:softprob",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
            "num_class": 26,
        }

    def get_num_classes(self):
        return 26

    def set_partitioner(self, partitioner: Partitioner) -> None:
        self.fds = FederatedDataset(
            dataset="mstz/covertype",
            subset="covertype",
            partitioners={"train": partitioner},
            resplitter=self.resplitter,
        )

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways', 'hillshade_9am', 'hillshade_noon', 'hillshade_3pm', 'horizontal_distance_to_fire_points', 'wilderness_area_id_0', 'wilderness_area_id_1', 'wilderness_area_id_2', 'wilderness_area_id_3', 'soil_type_id_0', 'soil_type_id_1', 'soil_type_id_2', 'soil_type_id_3', 'soil_type_id_4', 'soil_type_id_5', 'soil_type_id_6', 'soil_type_id_7', 'soil_type_id_8', 'soil_type_id_9', 'soil_type_id_10', 'soil_type_id_11', 'soil_type_id_12', 'soil_type_id_13', 'soil_type_id_14', 'soil_type_id_15', 'soil_type_id_16', 'soil_type_id_17', 'soil_type_id_18', 'soil_type_id_19', 'soil_type_id_20', 'soil_type_id_21', 'soil_type_id_22', 'soil_type_id_23', 'soil_type_id_24', 'soil_type_id_25', 'soil_type_id_26', 'soil_type_id_27', 'soil_type_id_28', 'soil_type_id_29', 'soil_type_id_30', 'soil_type_id_31', 'soil_type_id_32', 'soil_type_id_33', 'soil_type_id_34', 'soil_type_id_35', 'soil_type_id_36', 'soil_type_id_37', 'soil_type_id_38', 'soil_type_id_39'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['cover_type']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    