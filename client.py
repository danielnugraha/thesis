import warnings
from logging import INFO

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log

from thesis_dataset import (
    instantiate_partitioner,
)
from utils import client_args_parser, NUM_LOCAL_ROUND
from client_utils import XgbClient
from subsampling.mvs import MVS
from dataloader.multiclass_dataloader import CovertypeDataloader
from dataloader.binary_dataloader import HiggsDataloader
from dataloader.regression_dataloader import WineQualityDataloader


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
# Add arguments for subsampling strategy and dataset you want to use
args = client_args_parser()

# Train method (bagging or cyclic)
train_method = args.train_method

# Instantiate partitioner from ["uniform", "linear", "square", "exponential"]
partitioner = instantiate_partitioner(
    partitioner_type=args.partitioner_type, num_partitions=args.num_partitions
)

# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
dataloader = HiggsDataloader(partitioner)
train_dmatrix, num_train, = dataloader.get_train_dmatrix(node_id=args.partition_id)
valid_dmatrix, num_val = dataloader.get_test_dmatrix(node_id=args.partition_id if args.centralised_eval else None)

# Hyper-parameters for xgboost training
num_local_round = NUM_LOCAL_ROUND
params = dataloader.get_params()

# Setup learning rate
if args.train_method == "bagging" and args.scaled_lr:
    new_lr = params["eta"] / args.num_partitions
    params.update({"eta": new_lr})

# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=XgbClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        train_method,
        MVS(dataloader.get_objective(), sample_rate=0.2),
        args.visualise,
    ),
)

#Try global sampling but keep local as focus
#Visualise
#Proof that federated converge (faster)
#Add pyproject.toml
#Add adaptive from MVS 
