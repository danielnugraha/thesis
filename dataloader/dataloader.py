from typing import Optional, Union, Tuple, Any
import xgboost as xgb
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Dataloader:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def get_train_dmatrix(self, node_id: Optional[int] = None) -> tuple[xgb.DMatrix, int]:
        partition = self._get_partition(node_id, split="train")
        partition.set_format("numpy")
        train_data, _ = train_test_split(partition, test_fraction=0.2, seed=42)
        return self._transform_dataset_to_dmatrix(data=train_data), train_data.shape[0]
    
    def get_test_dmatrix(self,node_id: Optional[int] = None) -> tuple[xgb.DMatrix, int]:
        partition = self._get_partition(node_id, split="test")
        partition.set_format("numpy")
        _, test_data = train_test_split(partition, test_fraction=0.2, seed=42)
        return self._transform_dataset_to_dmatrix(data=test_data), test_data.shape[0]
    
    def dataset_analysis(self, node_id: Optional[int] = None):
        partition = self._get_partition(node_id, split="train")
        df = self._preprocess_to_pandas(partition)

        print(df.describe())
        print(df.info())
        print(df.head())

        fig, axs = self._create_analysis_subplots()
        axs = axs.flatten()

        # Plot histograms for each feature
        for i, col in enumerate(df.columns):
            sns.histplot(df[col], bins=30, kde=True, ax=axs[i])
            axs[i].set_title(f'Distribution of {col}')

        # Hide any remaining empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        # Plot the correlation matrix
        plt.figure(figsize=self._set_correlation_figure_size())
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'_static/{self.dataset_name}_correlation_matrix.png')
        plt.close()

        # Save all plots to a single PNG file
        fig.tight_layout()
        fig.savefig(f'_static/{self.dataset_name}_analysis.png')

    def get_objective(self):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def get_params(self):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def get_num_classes(self):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def _get_partition(self, node_id: Optional[int] = None, split: Optional[str] = None) -> Dataset:
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def _transform_dataset_to_dmatrix(self, data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
        raise NotImplementedError("This method needs to be implemented by subclasses")
    
    def _preprocess_to_pandas(self, partition: Dataset) -> pd.DataFrame:
        raise NotImplementedError("This method needs to be implemented by subclasses")
    
    def _create_analysis_subplots(self) -> Tuple[plt.Figure, Any]:
        raise NotImplementedError("This method needs to be implemented by subclasses")
    
    def _set_correlation_figure_size(self) -> Tuple[float, float]:
        raise NotImplementedError("This method needs to be implemented by subclasses")
    

def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    return partition_train, partition_test
