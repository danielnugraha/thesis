import numpy as np
import xgboost as xgb
from subsampling.subsampling_strategy import SubsamplingStrategy
from visualization import plot_tree, plot_labels
from flwr_datasets.partitioner import IidPartitioner
from typing import Optional, Tuple

class Random(SubsamplingStrategy):

    def __init__(self, objective, sample_rate) -> None:
        self.objective = objective
        self.sample_rate = sample_rate
        self.regularized_gradients_cache: Optional[np.ndarray] = None
        self.threshold = []

    def threshold_subsample(self, train_dmatrix: xgb.DMatrix, threshold: int) -> xgb.DMatrix:
        if self.regularized_gradients_cache is None:
            raise ValueError("No regularized gradients in memory.")
        
        indices = np.where(self.regularized_gradients_cache >= threshold)[0]
        new_train_dmatrix = train_dmatrix.slice(indices)
        return new_train_dmatrix

    def subsample(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> xgb.DMatrix:
        subsample_indices = np.random.choice(len(predictions), int(len(predictions) * self.sample_rate), replace=False)

        new_train_dmatrix = train_dmatrix.slice(subsample_indices)

        return new_train_dmatrix
    
    def subsample_indices(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix, grad_hess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        pass

    def grad_and_hess(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    def global_sampling(self, grad_hess_dict: dict[int, list[(float, float)]]) -> dict[int, list[int]]:
        sampling_values = {}
        all_gradients = []
        all_hessians = []

        for id, (grad, hess) in grad_hess_dict.items():
            all_gradients.extend(grad)
            all_hessians.extend(hess)

        all_gradients = np.array(all_gradients)
        all_hessians = np.array(all_hessians)

    def get_threshold(self) -> int:
        return self.threshold[-1]