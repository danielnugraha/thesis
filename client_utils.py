from logging import INFO
import xgboost as xgb
from subsampling.subsampling_strategy import SubsamplingStrategy
import flwr as fl
import csv
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from typing import Optional


class XgbClient(fl.client.Client):
    def __init__(
        self,
        train_dmatrix: xgb.DMatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        subsampling,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params 
        self.subsampling_method: Optional[SubsamplingStrategy] = subsampling

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input: xgb.Booster):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            if self.subsampling_method is None:
                bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())
            else:
                preds = bst_input.predict(self.train_dmatrix, training=True)
                important_features = bst_input.get_score(importance_type='gain')

                sorted_features = sorted(important_features.items(), key=lambda x: x[1], reverse=True)

                highest_2_features = sorted_features[:2]
                if len(highest_2_features) < 2:
                    x, y = 0, 1
                else:
                    x, y = tuple(int(key.replace('f', '')) for key, value in highest_2_features)
                new_train_dmatrix = self.subsampling_method.subsample(preds, self.train_dmatrix, x=x, y=y)
                bst_input.update(new_train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for server aggregation
        # Cyclic: return the entire model
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
        )

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.Booster(self.params, [self.train_dmatrix])
            bst = self._local_boost(bst)
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)
        byte_size = len(local_model_bytes)

        csv_file_name = f'_static/{global_round}.csv'
        with open(csv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([byte_size])

        print("model bytes size: ", len(local_model_bytes))
        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"AUC = {auc} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc},
        )
