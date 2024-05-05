from logging import INFO
import xgboost as xgb
from subsampling_strategy import SubsamplingStrategy
import flwr as fl
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
from visualization import plot_labels, plot_tree
import matplotlib.pyplot as plt
from dataloader import Dataloader


class XgbClient(fl.client.Client):
    def __init__(
        self,
        train_dmatrix: xgb.DMatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        train_method,
        subsampling_method: SubsamplingStrategy,
        dataloader: Dataloader,
        visualise: bool,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method
        self.subsampling_method = subsampling_method
        self.dataloader = dataloader
        self.visualise = visualise

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
            print(self.train_dmatrix.feature_names)
            preds = bst_input.predict(self.train_dmatrix, output_margin=True, training=True)
            new_train_dmatrix = self.subsampling_method.subsample(preds, self.train_dmatrix)
            bst_input.update(new_train_dmatrix, bst_input.num_boosted_rounds())

        print(bst_input.get_dump(dump_format="text"))

        # Bagging: extract the last N=num_local_round trees for server aggregation
        # Cyclic: return the entire model
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

        return bst
    
    def query_grad_and_hess(self, objective):
        bst = xgb.Booster(self.params, [self.train_dmatrix])
        preds = bst.predict(self.train_dmatrix, output_margin=True, training=True)
        return objective(preds, self.train_dmatrix)

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.Booster(self.params, [self.train_dmatrix])
            bst = self._local_boost(bst)
            print("first round")
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)
            bst.save_model(f"_static/model_{bst.num_boosted_rounds()}.json")

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)
        
        

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
