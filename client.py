import warnings
from logging import INFO

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log

from utils import (
    instantiate_partitioner, instantiate_dataloader, instantiate_sampling_method
)
from utils import generic_args_parser, NUM_LOCAL_ROUND
from client_utils import XgbClient
from subsampling.mvs import MVS
from dataloader.multiclass_dataloader import CovertypeDataloader
from dataloader.binary_dataloader import HiggsDataloader, JannisDataloader
from dataloader.regression_dataloader import WineQualityDataloader, HouseSalesDataloader, AllstateClaimsSeverityDataloader


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
# Add arguments for subsampling strategy and dataset you want to use
args = generic_args_parser()

# Train method (bagging or cyclic)
sample_rate = args.sample_rate
partitioner_type = args.partitioner_type
sampling_method = args.sampling_method

# Instantiate partitioner from ["uniform", "linear", "square", "exponential"]
partitioner = instantiate_partitioner(
    partitioner_type=args.partitioner_type, num_partitions=args.num_partitions
)

# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
dataloader = instantiate_dataloader(args.dataloader, partitioner)
# dataloader = CovertypeDataloader(partitioner)
train_dmatrix, num_train, = dataloader.get_train_dmatrix(node_id=args.partition_id)
valid_dmatrix, num_val = dataloader.get_test_dmatrix(node_id=args.partition_id)

# Hyper-parameters for xgboost training
num_local_round = NUM_LOCAL_ROUND
params = dataloader.get_params()

subsampling_strategy = instantiate_sampling_method(sampling_method, dataloader.get_objective(), sample_rate)
if subsampling_strategy is None:
    params.update({"subsample": sample_rate})

print(params)

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
        subsampling_strategy
    ),
)

#Try global sampling but keep local as focus
#Visualise
#Proof that federated converge (faster)
#Add pyproject.toml
#Add adaptive from MVS 
