from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.resplitter import Resplitter
from typing import Optional, Union, Dict, Tuple
import xgboost as xgb
import numpy as np
from objective import rmse_obj, rmsle_obj
from datasets import Dataset, DatasetDict, load_dataset
from dataloader.dataloader import Dataloader, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

class RegressionDataloader(Dataloader):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)

    def get_objective(self):
        return rmse_obj

    def get_params(self):
        return {
            "objective": "reg:squarederror",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "rmse",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return -1

class WineQualityDataloader(RegressionDataloader):
    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("wine_quality")
        self.fds = FederatedDataset(
            dataset="codesignal/wine-quality",
            partitioners={"white": partitioner},
        ) 
    
    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.fds.load_full("white")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="white")
        
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['quality']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    

class AllstateClaimsSeverityDataloader(RegressionDataloader):
    def __init__(self, partitioner: Partitioner) -> None:
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_cat/Allstate_Claims_Severity.csv")["train"]
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)

        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat111', 'cat114', 'cont1', 'cont2', 'cont3', 'cont4', 'cont5', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12', 'cont13', 'cont14'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['loss']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class HouseSalesDataloader(Dataloader):
    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_cat/house_sales.csv")["train"]
        partitioner.dataset = self.dataset
        self.partitioner = partitioner
        self.resplitter = resplitter    

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

    def get_objective(self):
        return rmse_obj
    
    def get_params(self):
        return {
            "objective": "reg:squarederror",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "rmse",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return -1

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'waterfront', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date_year', 'date_month', 'date_day'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['price']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class DiamondsDataloader(Dataloader):
    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_num/diamonds.csv")["train"]
        partitioner.dataset = self.dataset
        self.partitioner = partitioner
        self.resplitter = resplitter    

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

    def get_objective(self):
        return rmse_obj
    
    def get_params(self):
        return {
            "objective": "reg:squarederror",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "rmse",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return -1

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['carat', 'depth', 'table', 'x', 'y', 'z'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['price']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
