from dataloader.dataloader import Dataloader, train_test_split
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.resplitter import Resplitter
from typing import Optional, Union, Dict, Tuple
from datasets import Dataset, DatasetDict, load_dataset
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import arff
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
    
    def get_test_dmatrix(self, node_id: Optional[int] = None) -> tuple[xgb.DMatrix, int]:
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
            "max_depth": 10,
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
    
    def dataset_analysis(self, node_id: Optional[int] = None):
        if node_id is None:
            partition = self.fds.load_full("train")
        else:
            partition = self.fds.load_partition(node_id=node_id, split="train")
        
        df = partition.to_pandas()
        inputs_expanded = pd.DataFrame(df['inputs'].tolist(), columns=[f'feature_{i+1}' for i in range(28)])
        df = df.drop(columns=['inputs']).join(inputs_expanded)

        print(df.describe())
        print(df.info())
        print(df.head())

        # Plot distribution and correlation
        fig, axs = plt.subplots(7, 5, figsize=(20, 25))
        axs = axs.flatten()

        # Plot histograms for each feature
        for i, col in enumerate(inputs_expanded.columns):
            sns.histplot(df[col], bins=30, kde=True, ax=axs[i])
            axs[i].set_title(f'Distribution of {col}')

        # Hide any remaining empty subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        # Plot the correlation matrix
        plt.figure(figsize=(15, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('_static/higgs_correlation_matrix.png')
        plt.close()

        # Save all plots to a single PNG file
        fig.tight_layout()
        fig.savefig('_static/higgs_analysis.png')
    

class RoadSafetyDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="clf_cat/road-safety.csv")["train"]
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
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 10,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return 2

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['Vehicle_Reference_df_res', 'Vehicle_Type', 'Vehicle_Manoeuvre', 'Vehicle_Location-Restricted_Lane', 'Hit_Object_in_Carriageway', 'Hit_Object_off_Carriageway', 'Was_Vehicle_Left_Hand_Drive?', 'Age_of_Driver', 'Age_Band_of_Driver', 'Engine_Capacity_(CC)', 'Propulsion_Code', 'Age_of_Vehicle', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 'Longitude', 'Latitude', 'Police_Force', 'Number_of_Vehicles', 'Number_of_Casualties', 'Local_Authority_(District)', '1st_Road_Number', '2nd_Road_Number', 'Urban_or_Rural_Area', 'Vehicle_Reference_df', 'Casualty_Reference', 'Sex_of_Casualty', 'Age_of_Casualty', 'Age_Band_of_Casualty', 'Pedestrian_Location', 'Pedestrian_Movement', 'Casualty_Type', 'Casualty_IMD_Decile'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['SexofDriver']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
    def dataset_analysis(self, node_id: Optional[int] = None):
        if node_id is None:
            partition = self.dataset
        else:
            partition = self.partitioner.load_partition(node_id=node_id)
        
        df = partition.to_pandas()

        print(df.describe())
        print(df.info())
        print(df.head())
        
        # Plot distribution and correlation
        fig, axs = plt.subplots(7, 5, figsize=(20, 25))
        axs = axs.flatten()

        # Plot histograms for each feature
        for i, col in enumerate(df.columns):
            sns.histplot(df[col], bins=30, kde=True, ax=axs[i])
            axs[i].set_title(f'Distribution of {col}')

        # Hide any remaining empty subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        # Plot the correlation matrix
        plt.figure(figsize=(15, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('_static/road_safety_correlation_matrix.png')
        plt.close()

        # Save all plots to a single PNG file
        fig.tight_layout()
        fig.savefig('_static/road_safety_analysis.png')


class MiniBooNEDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        data = arff.loadarff('./dataloader/miniboone_dataset.arff')
        self.dataset = Dataset.from_pandas(pd.DataFrame(data[0]))
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
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 10,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return 2

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['ParticleID_0', 'ParticleID_1', 'ParticleID_2', 'ParticleID_3', 'ParticleID_4', 'ParticleID_5', 'ParticleID_6', 'ParticleID_7', 'ParticleID_8', 'ParticleID_9', 'ParticleID_10', 'ParticleID_11', 'ParticleID_12', 'ParticleID_13', 'ParticleID_14', 'ParticleID_15', 'ParticleID_16', 'ParticleID_17', 'ParticleID_18', 'ParticleID_19', 'ParticleID_20', 'ParticleID_21', 'ParticleID_22', 'ParticleID_23', 'ParticleID_24', 'ParticleID_25', 'ParticleID_26', 'ParticleID_27', 'ParticleID_28', 'ParticleID_29', 'ParticleID_30', 'ParticleID_31', 'ParticleID_32', 'ParticleID_33', 'ParticleID_34', 'ParticleID_35', 'ParticleID_36', 'ParticleID_37', 'ParticleID_38', 'ParticleID_39', 'ParticleID_40', 'ParticleID_41', 'ParticleID_42', 'ParticleID_43', 'ParticleID_44', 'ParticleID_45', 'ParticleID_46', 'ParticleID_47', 'ParticleID_48', 'ParticleID_49'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['signal']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
    def dataset_analysis(self, node_id: Optional[int] = None):
        if node_id is None:
            partition = self.dataset
        else:
            partition = self.partitioner.load_partition(node_id=node_id)
        
        df = partition.to_pandas()
        df['signal'] = df['signal'].apply(lambda x: x.decode('utf-8')).replace({'True': 1, 'False': 0})

        print(df.describe())
        print(df.info())
        print(df.head())
        
        # Plot distribution and correlation
        fig, axs = plt.subplots(10, 6, figsize=(20, 25))
        axs = axs.flatten()

        # Plot histograms for each feature
        for i, col in enumerate(df.columns):
            sns.histplot(df[col], bins=30, kde=True, ax=axs[i])
            axs[i].set_title(f'Distribution of {col}')

        # Hide any remaining empty subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        # Plot the correlation matrix
        plt.figure(figsize=(15, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('_static/mini_boone_arff_correlation_matrix.png')
        plt.close()

        # Save all plots to a single PNG file
        fig.tight_layout()
        fig.savefig('_static/mini_boone_arff_analysis.png')
    

class EyeMovementsDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="clf_cat/eye_movements.csv")["train"]
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
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 10,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return 2

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['lineNo', 'assgNo', 'P1stFixation', 'P2stFixation', 'prevFixDur', 'firstfixDur', 'firstPassFixDur', 'nextFixDur', 'firstSaccLen', 'lastSaccLen', 'prevFixPos', 'landingPos', 'leavingPos', 'totalFixDur', 'meanFixDur', 'regressLen', 'nextWordRegress', 'regressDur', 'pupilDiamMax', 'pupilDiamLag', 'timePrtctg', 'titleNo', 'wordNo'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['label']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class JannisDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/jannis.csv")["train"]
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
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 10,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return 2

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['class']
        new_data = xgb.DMatrix(x, label=y)
        return new_data


class CaliforniaDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        self.dataset = load_dataset("inria-soda/tabular-benchmark", data_files="clf_num/california.csv")["train"]
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
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 10,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return 2

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['price_above_median']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
    def dataset_analysis(self, node_id: Optional[int] = None):
        if node_id is None:
            partition = self.dataset
        else:
            partition = self.partitioner.load_partition(node_id=node_id)
        
        df = partition.to_pandas()

        print(df.describe())
        print(df.info())
        print(df.head())
        
        # Plot distribution and correlation
        fig, axs = plt.subplots(10, 6, figsize=(20, 25))
        axs = axs.flatten()

        # Plot histograms for each feature
        for i, col in enumerate(df.columns):
            sns.histplot(df[col], bins=30, kde=True, ax=axs[i])
            axs[i].set_title(f'Distribution of {col}')

        # Hide any remaining empty subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        # Plot the correlation matrix
        plt.figure(figsize=(15, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('_static/california_correlation_matrix.png')
        plt.close()

        # Save all plots to a single PNG file
        fig.tight_layout()
        fig.savefig('_static/california_analysis.png')
    

class AppetencyDataloader(Dataloader):

    def __init__(self, partitioner: Partitioner, resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]] = None,) -> None:
        data = arff.loadarff('./dataloader/appetency_dataset.arff')
        self.dataset = Dataset.from_pandas(pd.DataFrame(data[0]))
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
        return binary_obj

    def get_params(self):
        return {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 10,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }

    def get_num_classes(self):
        return 2

    def set_partitioner(self, partitioner: Partitioner) -> None:
        partitioner.dataset = self.dataset
        self.partitioner = partitioner

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        x_dict = data.with_format("np", ['ParticleID_0', 'ParticleID_1', 'ParticleID_2', 'ParticleID_3', 'ParticleID_4', 'ParticleID_5', 'ParticleID_6', 'ParticleID_7', 'ParticleID_8', 'ParticleID_9', 'ParticleID_10', 'ParticleID_11', 'ParticleID_12', 'ParticleID_13', 'ParticleID_14', 'ParticleID_15', 'ParticleID_16', 'ParticleID_17', 'ParticleID_18', 'ParticleID_19', 'ParticleID_20', 'ParticleID_21', 'ParticleID_22', 'ParticleID_23', 'ParticleID_24', 'ParticleID_25', 'ParticleID_26', 'ParticleID_27', 'ParticleID_28', 'ParticleID_29', 'ParticleID_30', 'ParticleID_31', 'ParticleID_32', 'ParticleID_33', 'ParticleID_34', 'ParticleID_35', 'ParticleID_36', 'ParticleID_37', 'ParticleID_38', 'ParticleID_39', 'ParticleID_40', 'ParticleID_41', 'ParticleID_42', 'ParticleID_43', 'ParticleID_44', 'ParticleID_45', 'ParticleID_46', 'ParticleID_47', 'ParticleID_48', 'ParticleID_49'])[:]
        x_arrays = list(x_dict.values())
        x = np.stack(x_arrays, axis=1)
        y = data['signal']
        new_data = xgb.DMatrix(x, label=y)
        return new_data
    
    def dataset_analysis(self, node_id: Optional[int] = None):
        if node_id is None:
            partition = self.dataset
        else:
            partition = self.partitioner.load_partition(node_id=node_id)
        
        df = partition.to_pandas()

        print(df.describe())
        print(df.info())
        print(df.head())
        
        # Plot distribution and correlation
        fig, axs = plt.subplots(10, 6, figsize=(20, 25))
        axs = axs.flatten()

        # Plot histograms for each feature
        for i, col in enumerate(df.columns):
            sns.histplot(df[col], bins=30, kde=True, ax=axs[i])
            axs[i].set_title(f'Distribution of {col}')

        # Hide any remaining empty subplots
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        # Plot the correlation matrix
        plt.figure(figsize=(15, 10))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('_static/appetency_correlation_matrix.png')
        plt.close()

        # Save all plots to a single PNG file
        fig.tight_layout()
        fig.savefig('_static/appetency_analysis.png')
