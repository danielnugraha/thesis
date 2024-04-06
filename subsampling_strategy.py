import numpy as np
import xgboost as xgb
from abc import ABC, abstractmethod
from typing import Optional

class SubsamplingStrategy(ABC):
    @property
    def dataset(self):
        pass
    
    @abstractmethod
    def subsample(self, bst: Optional[xgb.Booster] = None) -> xgb.Booster:
        pass

