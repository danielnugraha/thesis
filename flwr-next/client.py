from flwr.client import ClientApp
from flwr.common import Message, Context, RecordSet, ConfigsRecord, ParametersRecord, Array, MetricsRecord
from .xgb_client import XGBClientAdaptive
from .env import DATASET, SUBSAMPLING, NUM_LOCAL_ROUND, SAMPLE_RATE
from typing import Optional

app = ClientApp()

client = XGBClientAdaptive(
    NUM_LOCAL_ROUND,
    DATASET,
    SUBSAMPLING
)

@app.train()
def train(msg: Message, ctx: Context):
    threshold: Optional[int] = msg.content.configs_records["config"].get("threshold", None)
    partition_id: int = msg.content.configs_records["config"]["partition_id"]

    parameters = []
    record = msg.content.parameters_records.get("parameters", default=None)

    if record is not None:
        for key in list(record.keys()):
            parameters.append(record[key].data)

    if threshold is not None:
        parameters = client.threshold_train(threshold, partition_id)
    else:
        parameters = client.train(partition_id, parameters)

    recordset = RecordSet(
        parameters_records={"parameters": ParametersRecord({"01": Array("", [], "", parameters)})}
    )
    return msg.create_reply(recordset)


@app.evaluate()
def evaluate(msg: Message, ctx: Context):
    parameters = []
    record = msg.content.parameters_records["parameters"]
    for key in list(record.keys()):
        parameters.append(record[key].data)
    result, num = client.evaluate(parameters)
    reply = RecordSet(
        metrics_records={"metrics": MetricsRecord({"eval_result": result, "num_row": num})}
    )
    return msg.create_reply(reply)


@app.query()
def query(msg: Message, ctx: Context):
    parameters = []
    record = msg.content.parameters_records.get("parameters", default=None)
    partition_id: int = msg.content.configs_records["config"]["partition_id"]

    if record is not None:
        for key in list(record.keys()):
            parameters.append(record[key].data)
            
    threshold = float(client.query_threshold(partition_id, parameters))
    reply = RecordSet(
        configs_records={"config": ConfigsRecord({"threshold": threshold, "sample_rate": SAMPLE_RATE})},
    )
    return msg.create_reply(reply)
