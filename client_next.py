from flwr.client import ClientApp
from flwr.common import Message, Context, RecordSet, ConfigsRecord
from client_utils import XgbClient
from utils import client_args_parser

import warnings
from logging import INFO

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log

from thesis_dataset import (
    instantiate_partitioner,
)
from utils import client_args_parser, BST_PARAMS, NUM_LOCAL_ROUND, softprob_obj
from client_utils import XgbClient
from mvs import MVS
from dataloader import CovertypeDataloader
from flwr.common.recordset_compat import recordset_to_fitins, fitres_to_recordset, recordset_to_evaluateins, evaluateres_to_recordset

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
dataloader = CovertypeDataloader(partitioner)
train_dmatrix, num_train, = dataloader.get_train_dmatrix(node_id=args.partition_id)
valid_dmatrix, num_val = dataloader.get_test_dmatrix(node_id=args.partition_id if args.centralised_eval else None)

# Hyper-parameters for xgboost training
num_local_round = NUM_LOCAL_ROUND
params = BST_PARAMS

# Setup learning rate
if args.train_method == "bagging" and args.scaled_lr:
    new_lr = params["eta"] / args.num_partitions
    params.update({"eta": new_lr})

app = ClientApp()
client = XgbClient(
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
            train_method,
            MVS(softprob_obj)
        )

@app.train()
def train(msg: Message, ctx: Context):
    fit_res = client.fit(recordset_to_fitins(msg.content, True))
    return msg.create_reply(fitres_to_recordset(fit_res, False))


@app.evaluate()
def eval(msg: Message, ctx: Context):
    evaluate_res = client.evaluate(recordset_to_evaluateins(msg.content, True))
    return msg.create_reply(evaluateres_to_recordset(evaluate_res))


@app.query()
def query(msg: Message, ctx: Context):
    grad, hess = client.query_grad_and_hess(softprob_obj)
    reply = RecordSet(
        configs_records={"grad_and_hess": ConfigsRecord({"grad": grad, "hess": hess})},
    )
    return msg.create_reply(reply)