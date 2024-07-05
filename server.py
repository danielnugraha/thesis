import warnings
from logging import INFO
from flwr.common import FitRes, Parameters, Scalar
from typing import Dict, List, Tuple, Union, Optional, cast
import json
import flwr as fl
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr_datasets import FederatedDataset, partitioner
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
import numpy as np
from utils import generic_args_parser
import xgboost as xgb
import math
import os
import random

warnings.filterwarnings("ignore", category=UserWarning)

class CustomFedXgbBagging(FedXgbBagging):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using bagging."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate all the client trees
        global_model = self.global_model
        for _, fit_res in results:
            update = fit_res.parameters.tensors
            for bst in update:
                random_number = random.random()
                print(random_number)
                if random_number > 0.5:
                    global_model = aggregate(global_model, bst)

        self.global_model = global_model

        return (
            Parameters(tensor_type="", tensors=[cast(bytes, global_model)]),
            {},
        )


def aggregate(
    bst_prev_org: Optional[bytes],
    bst_curr_org: bytes,
) -> bytes:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev_org:
        return bst_curr_org

    # Get the tree numbers
    tree_num_prev, _ = _get_tree_nums(bst_prev_org)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr_org)

    bst_prev = json.loads(bytearray(bst_prev_org))
    bst_curr = json.loads(bytearray(bst_curr_org))

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    bst_prev_bytes = bytes(json.dumps(bst_prev), "utf-8")

    return bst_prev_bytes


def _get_tree_nums(xgb_model_org: bytes) -> Tuple[int, int]:
    xgb_model = json.loads(bytearray(xgb_model_org))
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num

# Parse arguments for experimental settings
args = generic_args_parser()
pool_size = args.pool_size
num_rounds = args.num_rounds
num_clients_per_round = args.num_clients_per_round
dataloader_str = args.dataloader
sample_rate = args.sample_rate
sampling_method = args.sampling_method
partitioner_type = args.partitioner_type

def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


evals = []
num_rounds_elapsed = 1


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    global num_rounds_elapsed, evals
    filtered_metrics = [(num, metrics) for num, metrics in eval_metrics if not math.isnan(metrics["AUC"])]
    total_num = sum([num for num, _ in filtered_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in filtered_metrics]) / total_num
    )

    evals.append(round(auc_aggregated, 4))

    if num_rounds_elapsed == num_rounds:
        print(evals)
        print("Writing to a file")
        with open(f'_static/{dataloader_str}_{sampling_method}_{partitioner_type}_{sample_rate}_{num_clients_per_round}.txt', 'w') as file:
            file.write(','.join(map(str, evals)))
        
        if sampling_method == 'mvs':
            print("writing model sizes to a file")
            model_sizes = []
            for i in range(1, num_rounds+1):
                file_path = f'_static/{i}.csv'
                data = np.loadtxt(file_path, delimiter=',')
                average_size = np.mean(data)
                model_sizes.append(round(average_size, 3))
                os.remove(file_path)

            with open(f'_static/{dataloader_str}_{sampling_method}_{partitioner_type}_{sample_rate}_{num_clients_per_round}_size.txt', 'w') as file:
                file.write(','.join(map(str, model_sizes)))
                
    num_rounds_elapsed += 1
    
    print("Aggregate evals: ", evals)
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated



def get_evaluate_fn(test_data, params):
    """Return a function for centralised evaluation."""

    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        # If at the first round, skip the evaluation
        global rate
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation
            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            
            evals[rate].append(auc)

            if server_round % 150 == 0:
                rate += 0.1
                evals[rate] = []
            
            log(INFO, f"Eval results: {evals}")

            return 0, {"AUC": auc}

    return evaluate_fn

print("Min available clients: ", pool_size)
print("Min fit clients: ", num_clients_per_round)


strategy = FedXgbBagging(
    evaluate_function=None,
    fraction_fit=(float(num_clients_per_round) / pool_size),
    min_fit_clients=num_clients_per_round,
    min_available_clients=pool_size,
    min_evaluate_clients=2,
    fraction_evaluate=1.0,
    on_evaluate_config_fn=eval_config,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=(
        evaluate_metrics_aggregation
    ),
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=None,
)
