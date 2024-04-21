import numpy as np
import xgboost as xgb
from abc import ABC, abstractmethod


class SubsamplingStrategy(ABC):
    
    @abstractmethod
    def subsample(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> xgb.DMatrix:
        pass
    
    @abstractmethod
    def global_sampling(self, grad_hess_dict: dict[int, list[(float, float)]]) -> dict[int, list[int]]:
        pass