import warnings
from logging import INFO

import flwr as fl
from flwr.common.logger import log
from flwr_datasets import FederatedDataset, partitioner
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from utils import server_args_parser
from server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
    CyclicClientManager,
)
from thesis_dataset import resplit, transform_dataset_to_dmatrix
from dataloader import CovertypeDataloader


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
args = server_args_parser()
train_method = args.train_method
pool_size = 1
num_rounds = 1
num_clients_per_round = 1
num_evaluate_clients = 1
centralised_eval = args.centralised_eval

print("Min available clients: ", pool_size)
print("Min fit clients: ", num_clients_per_round)
print("Min evaluate clients: ", num_evaluate_clients)

# Load centralised test set
if centralised_eval:
    dataloader = CovertypeDataloader(partitioner=partitioner.IidPartitioner(
                    num_partitions=20
                ))
    
    test_dmatrix, num_test = dataloader.get_test_dmatrix(None)


# Define strategy
if train_method == "bagging":
    # Bagging training
    strategy = FedXgbBagging(
        evaluate_function=get_evaluate_fn(test_dmatrix) if centralised_eval else None,
        fraction_fit=(float(num_clients_per_round) / pool_size),
        min_fit_clients=num_clients_per_round,
        min_available_clients=pool_size,
        min_evaluate_clients=num_evaluate_clients if not centralised_eval else 0,
        fraction_evaluate=1.0 if not centralised_eval else 0.0,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=(
            evaluate_metrics_aggregation if not centralised_eval else None
        ),
    )
else:
    # Cyclic training
    strategy = FedXgbCyclic(
        fraction_fit=1.0,
        min_available_clients=pool_size,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
        on_evaluate_config_fn=eval_config,
        on_fit_config_fn=fit_config,
    )

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
    client_manager=CyclicClientManager() if train_method == "cyclic" else None,
)