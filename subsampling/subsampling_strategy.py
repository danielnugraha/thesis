import numpy as np
import xgboost as xgb
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class SubsamplingStrategy(ABC):
    
    @abstractmethod
    def threshold_subsample(self, train_dmatrix: xgb.DMatrix, threshold: int) -> xgb.DMatrix:
        pass

    @abstractmethod
    def subsample(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix, indices: Optional[np.ndarray] = None, threshold: Optional[int] = None) -> xgb.DMatrix:
        pass

    @abstractmethod
    def subsample_indices(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix, grad_hess: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        pass

    @abstractmethod
    def grad_and_hess(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        pass
    
    @abstractmethod
    def global_sampling(self, grad_hess_dict: dict[int, list[(float, float)]]) -> dict[int, list[int]]:
        pass
    
    @abstractmethod
    def get_threshold(self) -> int:
        pass