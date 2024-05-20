from flwr.client import ClientApp
from flwr.common import Message, Context, RecordSet, ConfigsRecord, ParametersRecord, Array
from xgb_client import XGBClientAdaptive
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
from subsampling.mvs import MVS
from objective import binary_obj
from dataloader.binary_dataloader import HiggsDataloader
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
dataloader = HiggsDataloader(partitioner)
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
client = XGBClientAdaptive(
            num_local_round,
            args.partition_id,
            dataloader,
            MVS(binary_obj, sample_rate=1.0)
        )

@app.train()
def train(msg: Message, ctx: Context):
    threshold: int = msg.content.configs_records.get("adaptive_threshold")["threshold"]
    parameters = client.train(threshold)
    recordset = RecordSet(
        parameters_records={"parameters": ParametersRecord({"01": Array("", [], "", parameters)})}
    )
    return msg.create_reply(recordset)


@app.evaluate()
def eval(msg: Message, ctx: Context):
    evaluate_res = client.evaluate(recordset_to_evaluateins(msg.content, True))
    return msg.create_reply(evaluateres_to_recordset(evaluate_res))


@app.query()
def query(msg: Message, ctx: Context):
    if msg.content.configs_records.get("grad_and_hess") is not None:
        grad, hess = client.query_grad_and_hess(softprob_obj)
        reply = RecordSet(
            configs_records={"grad_and_hess": ConfigsRecord({"grad": grad, "hess": hess})},
        )
    else:
        parameters = []
        record = msg.content.parameters_records["parameters"]
        for key in list(record.keys()):
            parameters.append(record[key].data)
        threshold = client.query_threshold(parameters)
        reply = RecordSet(
            configs_records={"adaptive_threshold": ConfigsRecord({"threshold": threshold, "sample_rate": 0.1})},
        )
    return msg.create_reply(reply)