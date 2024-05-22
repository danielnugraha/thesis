from dataloader.dataloader import Dataloader
from subsampling.subsampling_strategy import SubsamplingStrategy

import xgboost as xgb
from typing import List, Optional, Tuple


class XGBClientAdaptive():
    def __init__(
        self,
        num_local_round: int,
        dataloader: Dataloader,
        subsampling_method: SubsamplingStrategy,
    ):
        self.train_dmatrix: Optional[xgb.DMatrix] = None
        self.valid_dmatrix: Optional[xgb.DMatrix] = None
        self.num_local_round = num_local_round
        self.dataloader = dataloader
        self.params = dataloader.get_params()
        self.subsampling_method = subsampling_method
        self.bst: Optional[xgb.Booster] = None

    def query_threshold(self, partition_id: int, parameters: List[bytes]) -> float:
        if self.train_dmatrix is None and self.valid_dmatrix is None:
            self.train_dmatrix, _, = self.dataloader.get_train_dmatrix(node_id=partition_id)
            self.valid_dmatrix, _ = self.dataloader.get_test_dmatrix(node_id=partition_id)

        if len(parameters)> 0:
            self.bst = xgb.Booster(params=self.params)

            for item in parameters:
                global_model = bytearray(item)

            self.bst.load_model(global_model)
        
        else:
            self.bst = xgb.Booster(self.params, [self.train_dmatrix])

        preds = self.bst.predict(self.train_dmatrix, output_margin=True, training=True)
        self.subsampling_method.subsample_indices(preds, self.train_dmatrix)

        return self.subsampling_method.get_threshold()

    def threshold_train(self, threshold: int, partition_id: int, parameters: Optional[List[bytes]] = None) -> bytes: 
        if self.train_dmatrix is None and self.valid_dmatrix is None:
            self.train_dmatrix, _, = self.dataloader.get_train_dmatrix(node_id=partition_id)
            self.valid_dmatrix, _ = self.dataloader.get_test_dmatrix(node_id=partition_id)

        print(self.train_dmatrix.num_row())

        if parameters is not None:
            self.bst = xgb.Booster(params=self.params)

            for item in parameters:
                global_model = bytearray(item)

            self.bst.load_model(global_model)
        else:
            self.bst = xgb.Booster(self.params, [self.train_dmatrix])

        subsample = self.subsampling_method.threshold_subsample(self.train_dmatrix, threshold)
        self.bst.update(subsample, self.bst.num_boosted_rounds())
        
        local_model = self.bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return local_model_bytes
    
    def train(self, partition_id: int, parameters: List[bytes]) -> bytes: 
        if self.train_dmatrix is None and self.valid_dmatrix is None:
            self.train_dmatrix, _, = self.dataloader.get_train_dmatrix(node_id=partition_id)
            self.valid_dmatrix, _ = self.dataloader.get_test_dmatrix(node_id=partition_id)

        print(self.train_dmatrix.num_row())

        if len(parameters)> 0:
            self.bst = xgb.Booster(params=self.params)

            for item in parameters:
                global_model = bytearray(item)

            self.bst.load_model(global_model)
        
        else:
            self.bst = xgb.Booster(self.params, [self.train_dmatrix])

        preds = self.bst.predict(self.train_dmatrix, output_margin=True, training=True)
        subsample = self.subsampling_method.subsample(preds, self.train_dmatrix)
        self.bst.update(subsample, self.bst.num_boosted_rounds())
        
        local_model = self.bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return local_model_bytes
    
    def evaluate(self, parameters: List[bytes]) -> Tuple[float, int]:
        self.bst = xgb.Booster(params=self.params)

        for item in parameters:
            global_model = bytearray(item)

        self.bst.load_model(global_model)
        
        eval_results = self.bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        
        return round(float(eval_results.split("\t")[1].split(":")[1]), 4), self.valid_dmatrix.num_row()
    