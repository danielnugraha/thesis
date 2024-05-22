from subsampling.mvs import MVS
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)
from dataloader.binary_dataloader import HiggsDataloader
from dataloader.regression_dataloader import WineQualityDataloader, HouseSalesDataloader
from dataloader.multiclass_dataloader import CovertypeDataloader

NUM_ROUNDS = 10
NUM_LOCAL_ROUND = 1
NUM_TRAIN_CLIENTS = 10
NUM_EVALUATE_CLIENTS = 10
CENTRALIZED_EVAL = True
PARTITIONER = IidPartitioner(num_partitions=NUM_TRAIN_CLIENTS)
DATASET = HouseSalesDataloader(PARTITIONER)
SAMPLE_RATE = 1.0
SUBSAMPLING = MVS(DATASET.get_objective(), sample_rate=SAMPLE_RATE)
