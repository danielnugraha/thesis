from typing import List, Tuple, Dict
import numpy as np
import xgboost as xgb
import json

import flwr as fl
from flwr.common import (
    Context,
    Array,
    MessageType,
    ParametersRecord,
    ConfigsRecord,
    RecordSet,
    DEFAULT_TTL,
)
from typing import Optional, List
from flwr.server import Driver
from .env import SUBSAMPLING, DATASET, NUM_TRAIN_CLIENTS, NUM_ROUNDS


subsampling = SUBSAMPLING
dataset = DATASET
num_clients = NUM_TRAIN_CLIENTS
num_rounds = NUM_ROUNDS

# Run via `flower-server-app server:app`
app = fl.server.ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    print("Starting XGBoost subsampling run")
    node_ids = []
    eval_results = []
    centralized_evals = []

    while len(node_ids) < num_clients:
        node_ids = driver.get_node_ids()
    
    global_model: Optional[bytes] = None
    valid_dmatrix, _ = dataset.get_test_dmatrix(node_id=None)

    for server_round in range(num_rounds):
        print(f"Commencing server round {server_round + 1}")

        global_model = adaptive_train_workflow(node_ids, server_round, driver, context, global_model)
        # global_model = train_workflow(node_ids, server_round, driver, context, global_model)

        if global_model is not None:
            context.state.parameters_records["parameters"] = ParametersRecord({"global_model": Array("", [], "", global_model)})
        eval_result = evaluate_workflow(node_ids, server_round, driver, context)

        centralized_eval = centralized_evaluate_workflow(valid_dmatrix, dataset.get_params(), global_model)
        centralized_evals.append(centralized_eval)
        if eval_result is None:
            return
        
        eval_results.append(eval_result)
    
    print("Eval results: ", eval_results)
    print("Centralized eval results: ", centralized_evals)


def train_workflow(node_ids: List[int], current_round: int, driver: Driver, context: Context, global_model: Optional[bytes] = None, recordset: Optional[RecordSet] = None) -> Optional[bytes]:
    out_messages = []
    for id, node_id in enumerate(node_ids):
        if recordset is None:
            record_set = RecordSet(
                parameters_records=context.state.parameters_records,
                configs_records={"config": ConfigsRecord()}
            )
        else:
            record_set = recordset

        record_set.configs_records["config"]["partition_id"] = id
        out_messages.append(
            driver.create_message(
                content=record_set,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
                group_id=str(current_round),
            )
        )

    in_messages = list(driver.send_and_receive(out_messages))
    num_failures = len([msg for msg in in_messages if msg.has_error()])

    if num_failures > 0:
        return None

    for message in in_messages:
        parameters = message.content.parameters_records['parameters']
        for parameter in parameters.values():
            global_model = aggregate(global_model, parameter.data)

    return global_model


def adaptive_train_workflow(node_ids: List[int], current_round: int, driver: Driver, context: Context, global_model: Optional[bytes] = None) -> Optional[bytes]:
    out_messages = []
    for id, node_id in enumerate(node_ids):
        record_set = RecordSet(
            # configs_records={"grad_and_hess": ConfigsRecord({"grad": 1, "hess": 1})},
            configs_records={"config": ConfigsRecord({"threshold": 1, "sample_rate": 1, "partition_id": id})},
            parameters_records=context.state.parameters_records,
        )
        out_messages.append(
            driver.create_message(
                content=record_set,
                message_type=MessageType.QUERY,
                dst_node_id=node_id,
                group_id=str(current_round),
                ttl=DEFAULT_TTL,
            )
        )

    in_messages = list(driver.send_and_receive(out_messages))
    num_failures = len([msg for msg in in_messages if msg.has_error()])

    if num_failures > 0:
        return None
    
    thresholds = []
    sample_rate = 0.0

    for message in in_messages:
        config = message.content.configs_records['config']
        threshold = config['threshold']
        thresholds.append(threshold)
        sample_rate = config['sample_rate']

    threshold = (np.max(thresholds) - np.min(thresholds)) * (1.0 - sample_rate) + np.min(thresholds)
    adaptive_threshold = ConfigsRecord({"threshold": threshold})

    recordset = RecordSet(
        configs_records={"config": adaptive_threshold},
    )

    return train_workflow(node_ids, current_round, driver, context, global_model, recordset)


def evaluate_workflow(node_ids: List[int], current_round: int, driver: Driver, context: Context) -> float:
    out_messages = []
    for id, node_id in enumerate(node_ids):
        record_set = RecordSet(
            parameters_records=context.state.parameters_records,
            configs_records={"config": ConfigsRecord({"partition_id": id})}
        )

        out_messages.append(
            driver.create_message(
                content=record_set,
                message_type=MessageType.EVALUATE,
                dst_node_id=node_id,
                group_id=str(current_round),
            )
        )

    in_messages = list(driver.send_and_receive(out_messages))
    num_failures = len([msg for msg in in_messages if msg.has_error()])

    if num_failures > 0:
        return None

    total_num = 0
    weighted_sum = 0
    for message in in_messages:
        metrics = message.content.metrics_records["metrics"]
        total_num += metrics["num_row"]
        weighted_sum += metrics["eval_result"] * metrics["num_row"]

    return weighted_sum / total_num


def centralized_evaluate_workflow(valid_dmatrix: xgb.DMatrix, params, global_model: Optional[bytes]) -> float:
    bst = xgb.Booster(params=params)

    if global_model is None:
        return
    
    bst.load_model(bytearray(global_model))

    eval_result = bst.eval_set(
        evals=[(valid_dmatrix, "valid")],
        iteration=bst.num_boosted_rounds() - 1,
    )

    return round(float(eval_result.split("\t")[1].split(":")[1]), 4)


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


def global_sampling_workflow():
    #grad_hess_dict = {}

        #for id, msg in all_replies_dict.items():
         #   values = msg.content.configs_records["grad_and_hess"]
          #  grad: List[float] = values["grad"]
           # hess: List[float] = values["hess"]
            #grad_hess_dict[id] = (grad, hess)
        
        #configs_dict = subsampling.global_sampling(grad_hess_dict)
    pass
