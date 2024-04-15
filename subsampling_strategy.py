import numpy as np
import xgboost as xgb
from abc import ABC, abstractmethod


class SubsamplingStrategy(ABC):
    
    @abstractmethod
    def subsample(self, predictions: np.ndarray, train_dmatrix: xgb.DMatrix) -> xgb.DMatrix:
        pass
