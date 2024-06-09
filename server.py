import warnings
from logging import INFO
from flwr.common import Parameters, Scalar
from typing import Dict
import json
import flwr as fl
from flwr.common.logger import log
from flwr_datasets import FederatedDataset, partitioner
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from utils import generic_args_parser
import xgboost as xgb

warnings.filterwarnings("ignore", category=UserWarning)

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
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )

    evals.append(round(auc_aggregated, 4))

    if num_rounds_elapsed == num_rounds:
        print(evals)
        print("Writing to a file")
        with open(f'_static/{dataloader_str}_{sampling_method}_{partitioner_type}_{sample_rate}_{num_clients_per_round}.txt', 'w') as file:
            file.write(','.join(map(str, evals)))
                
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
