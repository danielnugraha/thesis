from dataloader.dataloader import Dataloader, train_test_split
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.resplitter import Resplitter
from typing import Optional, Union, Dict, Tuple, Any
from datasets import Dataset, DatasetDict
import xgboost as xgb
import numpy as np
import openml
from objective import softprob_obj
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

class MulticlassDataloader(Dataloader):
    def __init__(self, dataset_name: str, num_classes: int) -> None:
        super().__init__(dataset_name)
        self.num_classes = num_classes

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
            "num_class": self.num_classes,
        }

    def get_num_classes(self):
        return self.num_classes


class CovertypeDataloader(MulticlassDataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("covertype", 7)
        self.fds = FederatedDataset(
            dataset="mstz/covertype",
            subset="covertype",
            partitioners={"train": partitioner},
        )

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.fds.load_full("train")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")

        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology', 'horizontal_distance_to_roadways', 'hillshade_9am', 'hillshade_noon', 'hillshade_3pm', 'horizontal_distance_to_fire_points', 'wilderness_area_id_0', 'wilderness_area_id_1', 'wilderness_area_id_2', 'wilderness_area_id_3', 'soil_type_id_0', 'soil_type_id_1', 'soil_type_id_2', 'soil_type_id_3', 'soil_type_id_4', 'soil_type_id_5', 'soil_type_id_6', 'soil_type_id_7', 'soil_type_id_8', 'soil_type_id_9', 'soil_type_id_10', 'soil_type_id_11', 'soil_type_id_12', 'soil_type_id_13', 'soil_type_id_14', 'soil_type_id_15', 'soil_type_id_16', 'soil_type_id_17', 'soil_type_id_18', 'soil_type_id_19', 'soil_type_id_20', 'soil_type_id_21', 'soil_type_id_22', 'soil_type_id_23', 'soil_type_id_24', 'soil_type_id_25', 'soil_type_id_26', 'soil_type_id_27', 'soil_type_id_28', 'soil_type_id_29', 'soil_type_id_30', 'soil_type_id_31', 'soil_type_id_32', 'soil_type_id_33', 'soil_type_id_34', 'soil_type_id_35', 'soil_type_id_36', 'soil_type_id_37', 'soil_type_id_38', 'soil_type_id_39'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['cover_type']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    

class HelenaDataloader(MulticlassDataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("helena", 100)
        data, _, _, _ = openml.datasets.get_dataset(41169).get_data()
        data['class'] = data['class'].astype('object')
        self.dataset = Dataset.from_pandas(data)
        self.partitioner = partitioner

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['class']
        y_int = np.array([int(element.decode('utf-8')) for element in y])
        new_data = xgb.DMatrix(x, label=y_int)
        return new_data


class DionisDataloader(MulticlassDataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("helena", 355)
        data, _, _, _ = openml.datasets.get_dataset(41167).get_data()
        data['class'] = data['class'].astype('object')
        self.dataset = Dataset.from_pandas(data)
        self.partitioner = partitioner

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        
        return partition

    def get_train_dmatrix(self, node_id: Optional[int] = None) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        
        partition.set_format("numpy")
        train_data, _ = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=train_data), train_data.shape[0]
    
    def get_test_dmatrix(self, node_id: Optional[int]) -> tuple[xgb.DMatrix, int]:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        
        partition.set_format("numpy")
        _, test_data = train_test_split(
            partition, test_fraction=0.2, seed=42
        )
        
        return self._transform_dataset_to_dmatrix(data=test_data), test_data.shape[0]

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        x_T = x.T
        cols_to_keep = np.array([i for i in range(x_T.shape[0]) if len(np.unique(x_T[i])) > 1])
        filtered_x = x[:, cols_to_keep]
        y = data['class']
        y_int = np.array([int(element) for element in y])
        new_data = xgb.DMatrix(filtered_x, label=y_int)
        return new_data
    
    def _create_analysis_subplots(self) -> Tuple[plt.Figure, Any]:
        return plt.subplots(11, 6, figsize=(20, 25))
    
    def _set_correlation_figure_size(self) -> Tuple[float, float]:
        return (30, 20)
