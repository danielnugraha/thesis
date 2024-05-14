from typing import List, Tuple, Dict
import random
import time
import json

import flwr as fl
from flwr.common import (
    Context,
    NDArrays,
    Message,
    MessageType,
    Metrics,
    MetricsRecord,
    ConfigsRecord,
    RecordSet,
    DEFAULT_TTL,
)
from typing import Optional
from flwr.server import Driver
from subsampling.mvs import MVS
from objective import binary_obj


# Run via `flower-server-app server:app`
app = fl.server.ServerApp()
subsampling = MVS(binary_obj)


@app.main()
def main(driver: Driver, context: Context) -> None:
    """This is a stub example that simply sends and receives messages."""
    print("Starting test run")
    for server_round in range(3):
        print(f"Commencing server round {server_round + 1}")

        # Get node IDs
        node_ids = driver.get_node_ids()

        recordset = RecordSet(
            configs_records={"grad_and_hess": ConfigsRecord({"grad": 1, "hess": 1})},
        )

        for node_id in node_ids:
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.QUERY,
                dst_node_id=node_id,
                group_id=str(server_round),
                ttl=DEFAULT_TTL,
            )
            messages.append(message)

         # Send messages
        message_ids = driver.push_messages(messages)
        print(f"Pushed {len(message_ids)} messages: {message_ids}")

        # Wait for results, ignore empty message_ids
        message_ids = [message_id for message_id in message_ids if message_id != ""]
        all_replies: List[Message] = []
        while True:
            replies = driver.pull_messages(message_ids=message_ids)
            print(f"Got {len(replies)} results")
            all_replies += replies
            if len(all_replies) == len(message_ids):
                break
            time.sleep(3)

        print(f"Received {len(all_replies)} results")

        all_replies_dict = {msg.metadata.src_node_id: msg for msg in all_replies}
        grad_hess_dict = {}

        for id, msg in all_replies_dict.items():
            values = msg.content.configs_records["grad_and_hess"]
            grad: List[float] = values["grad"]
            hess: List[float] = values["hess"]
            grad_hess_dict[id] = (grad, hess)
        
        configs_dict = subsampling.global_sampling(grad_hess_dict)

        # Create messages
        recordset = RecordSet()
        messages = []
        for node_id in node_ids:
            recordset = RecordSet(
                configs_records={"data_points": ConfigsRecord({"indices": configs_dict[node_id]})},
            )
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
                group_id=str(server_round),
                ttl=DEFAULT_TTL,
            )
            messages.append(message)

        # Send messages
        message_ids = driver.push_messages(messages)
        print(f"Pushed {len(message_ids)} messages: {message_ids}")

        # Wait for results, ignore empty message_ids
        message_ids = [message_id for message_id in message_ids if message_id != ""]
        all_replies: List[Message] = []
        while True:
            replies = driver.pull_messages(message_ids=message_ids)
            print(f"Got {len(replies)} results")
            all_replies += replies
            if len(all_replies) == len(message_ids):
                break
            time.sleep(3)

        print(f"Received {len(all_replies)} results")

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
