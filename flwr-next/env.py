from subsampling.mvs import MVS
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from dataloader.binary_dataloader import HiggsDataloader
from dataloader.regression_dataloader import WineQualityDataloader, HouseSalesDataloader, AllstateClaimsSeverityDataloader
from dataloader.multiclass_dataloader import CovertypeDataloader

NUM_ROUNDS = 10
NUM_LOCAL_ROUND = 1
NUM_TRAIN_CLIENTS = 3
NUM_EVALUATE_CLIENTS = 3
CENTRALIZED_EVAL = True
PARTITIONER = IidPartitioner(num_partitions=NUM_TRAIN_CLIENTS)
DATASET = WineQualityDataloader(PARTITIONER)
SAMPLE_RATE = 1.0
SUBSAMPLING = MVS(DATASET.get_objective(), sample_rate=SAMPLE_RATE)


# 3, 5, 10, 20 clients
# MVS Adaptive, Goss, MVS subsampling 0.1 - 1.0
# Centralized eval and decentralized eval
# Linear and Uniform Partitioner (Exponential or Square)
# produce the graphs and store it together
