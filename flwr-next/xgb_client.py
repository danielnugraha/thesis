from dataloader.dataloader import Dataloader
from subsampling.subsampling_strategy import SubsamplingStrategy

import xgboost as xgb
from typing import List, Optional


class XGBClientAdaptive():
    def __init__(
        self,
        num_local_round: int,
        node_id: int,
        dataloader: Dataloader,
        subsampling_method: SubsamplingStrategy,
    ):
        train_dmatrix, num_train, = dataloader.get_train_dmatrix(node_id=node_id)
        valid_dmatrix, num_val = dataloader.get_test_dmatrix(node_id=node_id)
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = dataloader.get_params()
        self.subsampling_method = subsampling_method
        self.bst: Optional[xgb.Booster] = None

    def query_threshold(self, parameters: List[bytes]) -> int:
        bst = xgb.Booster(params=self.params)

        for item in parameters:
            global_model = bytearray(item)

        bst.load_model(global_model)

        preds = bst.predict(self.train_dmatrix, output_margin=True, training=True)
        self.subsampling_method.subsample_indices(preds, self.train_dmatrix)

        return self.subsampling_method.get_threshold()

    def train(self, threshold: int, parameters: Optional[List[bytes]] = None) -> bytes: 

        if parameters is not None:
            self.bst = xgb.Booster(params=self.params)

            for item in parameters:
                global_model = bytearray(item)

            self.bst.load_model(global_model)

        if self.bst is None:
            raise ValueError("XGBoost is not initialized")

        subsample = self.subsampling_method.threshold_subsample(self.train_dmatrix, threshold)
        self.bst.update(subsample, self.bst.num_boosted_rounds())
        
        local_model = self.bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return local_model_bytes
    