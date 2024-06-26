from utils import generic_args_parser, instantiate_dataloader
from flwr_datasets.partitioner import ExponentialPartitioner

args = generic_args_parser()
num_clients = 5
partitioner = ExponentialPartitioner(num_clients)
dataloader = instantiate_dataloader(args.dataloader, partitioner)
dataloader.dataset_analysis()
