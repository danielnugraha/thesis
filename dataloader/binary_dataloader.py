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
import openml
from sklearn.preprocessing import LabelEncoder

class BinaryDataloader(Dataloader):
    def __init__(self, dataset_name: str) -> None:
        super().__init__(dataset_name)

    def get_objective(self):
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 6,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "alpha": 3,
            "gamma": 3,
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
        x = np.delete(x, [2,4,7,11,15,19], axis=1)
        y = data["label"]
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
    def _preprocess_to_pandas(self, partition: Dataset) -> pd.DataFrame:
        df = partition.to_pandas()
        inputs_expanded = pd.DataFrame(df['inputs'].tolist(), columns=[f'feature_{i+1}' for i in range(28)])
        df = df.drop(columns=['inputs']).join(inputs_expanded)
        columns_to_delete = [2, 4, 7, 11, 15, 19]
        columns_to_delete = [f'feature_{i+1}' for i in columns_to_delete]
        df = df.drop(columns=columns_to_delete)
        return df
    

class RoadSafetyDataloader(BinaryDataloader):

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


class JannisDataloader(BinaryDataloader):

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
        x_dict = data.with_format("np", ['V3', 'V4', 'V6', 'V9', 'V12', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        print(x.shape)
        y = data['class']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
    def _preprocess_to_pandas(self, partition: Dataset) -> pd.DataFrame:
        df = partition.to_pandas()
        columns_to_drop = ['V1', 'V2', 'V5', 'V7', 'V8', 'V10', 'V20', 'V11', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V21', 'V22', 'V23', 'V24','V25', 'V26',]
        df.drop(columns=columns_to_drop, inplace=True)
        return df
    
    def _create_analysis_subplots(self) -> Tuple[plt.Figure, Any]:
        return plt.subplots(10, 6, figsize=(20, 25))
    
    def _set_correlation_figure_size(self) -> Tuple[float, float]:
        return (30, 20)
    
    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.01,  # Learning rate
            "max_depth": 6,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }


class CompasDataloader(BinaryDataloader):
    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("jannis")
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="clf_cat/compas-two-years.csv")["train"]
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['sex', 'age', 'juv_misd_count', 'priors_count', 'age_cat_25-45', 'age_cat_Greaterthan45', 'age_cat_Lessthan25', 'race_African-American', 'race_Caucasian', 'c_charge_degree_F', 'c_charge_degree_M'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['twoyearrecid']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    

class AirlinesDataloader(BinaryDataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("airlines")
        data, _, _, _ = openml.datasets.get_dataset(41672).get_data()
        data['Airline'] = data['Airline'].astype(str)
        data['AirportFrom'] = data['AirportFrom'].astype(str)
        data['AirportTo'] = data['AirportTo'].astype(str)
        data['DayOfWeek'] = data['DayOfWeek'].astype(str)
        data['Delay'] = data['Delay'].astype(int)
        self.dataset = Dataset.from_pandas(data)
        self.partitioner = partitioner
        self.partitioner.dataset = self.dataset

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        categorical_columns = ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek']
        encoders = {col: LabelEncoder() for col in categorical_columns}
        x_dict = data.with_format("np", ['Airline', 'Flight', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Time', 'Length'])[:]
        for col in categorical_columns:
            x_dict[col] = encoders[col].fit_transform(x_dict[col])

        x_arrays = [x_dict[col] for col in x_dict.keys()]
        x = np.column_stack(x_arrays)
        y = data['Delay']
        y_int = np.array([int(element) for element in y])
        new_data = xgb.DMatrix(x, label=y_int)
        print(new_data.get_data()[0])
        return new_data
    

class FairJobDataloader(BinaryDataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("road_safety")
        self.dataset = load_dataset("criteo/FairJob")["train"]
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


class AbaloneDataloader(BinaryDataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("abalone")
        self.dataset = load_dataset("mstz/abalone", "binary")["train"]
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight'])[:]
        x_arrays = list(x_dict.values())
        encoded = []
        for element in x_arrays[0]:
            if element == 'M': 
                encoded.append(0)
            if element == 'I':
                encoded.append(1)
            if element == 'F':
                encoded.append(2)
        x_arrays[0] = encoded
        x = np.stack(x_arrays, axis=1)
        y = data['is_old']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class CustomerChurnDataloader(BinaryDataloader):
    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("customer_churn")
        self.fds = FederatedDataset(
            dataset="aai510-group1/telco-customer-churn",
            partitioners={"train": partitioner, "test": partitioner},
        )

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.fds.load_full(split)
        else:
            partition = self.fds.load_partition(node_id=node_id, split=split)

        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        print(type(data))
        x_dict = data.with_format("np", ['Age', 'Avg Monthly GB Download', 'Avg Monthly Long Distance Charges', 'Churn Score', 'City', 'CLTV', 'Contract', 'Customer Status', 'Dependents', 'Device Protection Plan', 'Gender', 'Internet Service', 'Lat Long', 'Latitude', 'Longitude', 'Married', 'Monthly Charge', 'Multiple Lines', 'Number of Dependents', 'Number of Referrals', 'Online Backup', 'Online Security', 'Paperless Billing', 'Partner', 'Payment Method', 'Phone Service', 'Population', 'Premium Tech Support', 'Referred a Friend', 'Satisfaction Score', 'Senior Citizen', 'Streaming Movies', 'Streaming Music', 'Streaming TV', 'Tenure in Months', 'Total Charges', 'Total Extra Data Charges', 'Total Long Distance Charges', 'Total Refunds', 'Total Revenue', 'Under 30', 'Unlimited Data'])[:]
        x_arrays = list(x_dict.values())
        print(x_arrays)
        x = np.stack(x_arrays, axis=1)
        y = data['Churn']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class CodrnaDataloader(BinaryDataloader):

    def __init__(self, partitioner: Partitioner) -> None:
        super().__init__("airlines")
        data, _, _, _ = openml.datasets.get_dataset(351).get_data()
        for col in data.columns:
            if pd.api.types.is_sparse(data[col]):
                data[col] = data[col].sparse.to_dense()
        self.dataset = Dataset.from_pandas(data)
        self.partitioner = partitioner
        self.partitioner.dataset = self.dataset

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        if node_id is None:
            partition = self.dataset
        else:
            partition = partition = self.partitioner.load_partition(node_id=node_id)
        
        return partition

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['Y']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
