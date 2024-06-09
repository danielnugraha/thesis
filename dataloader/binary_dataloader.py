from dataloader.dataloader import Dataloader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from typing import Optional, Union, Tuple, Any
from datasets import Dataset, DatasetDict, load_dataset
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from objective import binary_obj

class BinaryDataloader(Dataloader):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)

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


class HiggsDataloader(BinaryDataloader):
    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("higgs")
        self.fds = FederatedDataset(
            dataset="jxie/higgs",
            partitioners={"train": partitioner},
        )

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.fds.load_full("train")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")

        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x = data["inputs"]
        y = data["label"]
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
    def _preprocess_to_pandas(self, partition: Dataset) -> pd.DataFrame:
        df = partition.to_pandas()
        inputs_expanded = pd.DataFrame(df['inputs'].tolist(), columns=[f'feature_{i+1}' for i in range(28)])
        df = df.drop(columns=['inputs']).join(inputs_expanded)
        return df
    

class RoadSafetyDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("road_safety")
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="clf_cat/road-safety.csv")["train"]
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['Vehicle_Reference_df_res', 'Vehicle_Type', 'Vehicle_Manoeuvre', 'Vehicle_Location-Restricted_Lane', 'Hit_Object_in_Carriageway', 'Hit_Object_off_Carriageway', 'Was_Vehicle_Left_Hand_Drive?', 'Age_of_Driver', 'Age_Band_of_Driver', 'Engine_Capacity_(CC)', 'Propulsion_Code', 'Age_of_Vehicle', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude', 'Police_Force', 'Number_of_Vehicles', 'Number_of_Casualties', 'Local_Authority_(District)', '1st_Road_Number', '2nd_Road_Number', 'Urban_or_Rural_Area', 'Vehicle_Reference_df', 'Casualty_Reference', 'Sex_of_Casualty', 'Age_of_Casualty', 'Age_Band_of_Casualty', 'Pedestrian_Location', 'Pedestrian_Movement', 'Casualty_Type', 'Casualty_IMD_Decile'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['SexofDriver']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class JannisDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("jannis")
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/jannis.csv")["train"]
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['class']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
