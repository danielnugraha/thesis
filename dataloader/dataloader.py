from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner, IidPartitioner
from flwr_datasets.resplitter import Resplitter
from typing import Optional, Union, Dict, Tuple
import xgboost as xgb
import numpy as np
from utils import softprob_obj, binary_obj, rmse_obj
from datasets import Dataset, DatasetDict, load_dataset
from abc import ABC, abstractmethod

class Dataloader(ABC):
    
    @abstractmethod
    def get_train_dmatrix(self, node_id: Optional[int]) -> tuple[xgb.DMatrix, int]:
        pass
    
    @abstractmethod
    def get_test_dmatrix(self,node_id: Optional[int]) -> tuple[xgb.DMatrix, int]:
        pass

    @abstractmethod
    def get_objective(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def get_num_classes(self):
        pass
    

def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    return partition_train, partition_test
    
    
class AdultDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.fds = FederatedDataset(
            dataset="mstz/adult",
            subset="income",
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
        pass

    def set_partitioner(self, partitioner: Partitioner) -> None:
        self.fds = FederatedDataset(
            dataset="mstz/adult",
            subset="income",
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

class AbaloneDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.fds = FederatedDataset(
            dataset="mstz/abalone",
            subset="abalone",
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
        pass

    def set_partitioner(self, partitioner: Partitioner) -> None:
        self.fds = FederatedDataset(
            dataset="mstz/abalone",
            subset="abalone",
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

class YearPredictionMSDDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.dataset = load_dataset("text", data_files={"train": "https://www.dropbox.com/scl/fi/gh9b5sqcy7bkujxs8z6uu/YearPredictionMSD.txt?rlkey=8rto7705uxpt5jwj81xn4p4wm&dl=1"})
        print(self.dataset["train"][1])
        print(len(self.dataset["train"]))
        self.partitioner = partitioner
        self.resplitter = resplitter

    def get_train_dmatrix(self, node_id: Optional[int] = None) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.dataset
        else:
            self.partitioner.dataset = self.dataset
            partition = self.partitioner.load_partition(node_id=node_id, split="train")
        
        partition.set_format("numpy")
        train_data, _ = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=train_data), train_data.shape[0]
    
    def get_test_dmatrix(self, node_id: Optional[int]) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.dataset
        else:
            self.partitioner.dataset = self.dataset
            partition = self.partitioner.load_partition(node_id=node_id, split="train")
        
        partition.set_format("numpy")
        _, test_data = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=test_data), test_data.shape[0]

    def get_objective(self):
        return rmse_obj
    
    def get_params(self):
        pass

    def get_num_classes(self):
        pass

    def set_partitioner(self, partitioner: Partitioner) -> None:
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways', 'hillshade_9am', 'hillshade_noon', 'hillshade_3pm', 'horizontal_distance_to_fire_points', 'wilderness_area_id_0', 'wilderness_area_id_1', 'wilderness_area_id_2', 'wilderness_area_id_3', 'soil_type_id_0', 'soil_type_id_1', 'soil_type_id_2', 'soil_type_id_3', 'soil_type_id_4', 'soil_type_id_5', 'soil_type_id_6', 'soil_type_id_7', 'soil_type_id_8', 'soil_type_id_9', 'soil_type_id_10', 'soil_type_id_11', 'soil_type_id_12', 'soil_type_id_13', 'soil_type_id_14', 'soil_type_id_15', 'soil_type_id_16', 'soil_type_id_17', 'soil_type_id_18', 'soil_type_id_19', 'soil_type_id_20', 'soil_type_id_21', 'soil_type_id_22', 'soil_type_id_23', 'soil_type_id_24', 'soil_type_id_25', 'soil_type_id_26', 'soil_type_id_27', 'soil_type_id_28', 'soil_type_id_29', 'soil_type_id_30', 'soil_type_id_31', 'soil_type_id_32', 'soil_type_id_33', 'soil_type_id_34', 'soil_type_id_35', 'soil_type_id_36', 'soil_type_id_37', 'soil_type_id_38', 'soil_type_id_39'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['cover_type']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
# CovertypeDataloader(partitioner=IidPartitioner(10)).get_train_dmatrix()   
# YearPredictionMSDDataloader(partitioner=IidPartitioner(10))